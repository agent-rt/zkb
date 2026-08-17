/* A custom FTS5 tokenizer: CJK bigrams + Latin whole words.
 *
 * Why this exists: FTS5's built-in options are all wrong for a bilingual corpus.
 * `unicode61` does not segment CJK at all (a whole Chinese paragraph becomes one
 * token, so nothing matches). `trigram` does match CJK, but by character
 * 3-grams — measured recall@10 on Chinese queries was 0.167, because unrelated
 * words sharing a 3-gram collide and BM25's ordering ends up near random
 * (docs/experiments/E2-baseline.md).
 *
 * Bigrams are the standard dictionary-free approach for Chinese: the majority of
 * Chinese words are two characters, so a bigram is usually a whole word rather
 * than a fragment of one. Overlapping bigrams also preserve phrase semantics —
 * "第一性" tokenizes to "第一","一性" at consecutive positions, so an FTS5 phrase
 * query still requires them adjacent in the document.
 *
 * Latin runs are emitted as whole words with ASCII case folding, which is what
 * makes `lopdf` or `SurrealDB` an exact match instead of a bag of 3-grams.
 *
 * The same algorithm runs for documents and for queries. That symmetry is the
 * whole correctness argument: a two-character query produces exactly the bigram
 * that indexing produced.
 *
 * The amalgamation's sqlite3.h already declares fts5_api / fts5_tokenizer, so
 * the real declarations are used rather than a hand-written prefix.
 */

#include "sqlite3.h"
#include <string.h>

#define ZKB_MAX_TOKEN 64

/* ---------------------------------------------------------------- UTF-8 ---- */

/* Decode one codepoint. Returns bytes consumed, or 1 for invalid input (which
 * advances rather than stalling — malformed bytes become separators). */
static int zkb_decode(const unsigned char *p, int n, unsigned int *out) {
    unsigned char c = p[0];
    if (c < 0x80) { *out = c; return 1; }
    if ((c & 0xE0) == 0xC0 && n >= 2) {
        *out = ((unsigned int)(c & 0x1F) << 6) | (p[1] & 0x3F);
        return 2;
    }
    if ((c & 0xF0) == 0xE0 && n >= 3) {
        *out = ((unsigned int)(c & 0x0F) << 12) | ((unsigned int)(p[1] & 0x3F) << 6) |
               (p[2] & 0x3F);
        return 3;
    }
    if ((c & 0xF8) == 0xF0 && n >= 4) {
        *out = ((unsigned int)(c & 0x07) << 18) | ((unsigned int)(p[1] & 0x3F) << 12) |
               ((unsigned int)(p[2] & 0x3F) << 6) | (p[3] & 0x3F);
        return 4;
    }
    *out = 0xFFFD;
    return 1;
}

/* Scripts written without spaces, and therefore segmented by bigram.
 * Kana and Hangul are included for the same reason as Han. */
static int zkb_is_cjk(unsigned int cp) {
    return (cp >= 0x3040 && cp <= 0x30FF)    /* Hiragana + Katakana        */
        || (cp >= 0x3400 && cp <= 0x4DBF)    /* CJK Ext A                  */
        || (cp >= 0x4E00 && cp <= 0x9FFF)    /* CJK Unified Ideographs     */
        || (cp >= 0xF900 && cp <= 0xFAFF)    /* CJK Compatibility          */
        || (cp >= 0xAC00 && cp <= 0xD7AF)    /* Hangul Syllables           */
        || (cp >= 0x20000 && cp <= 0x2FA1F); /* CJK Ext B and beyond       */
}

/* Characters that belong inside a word token. ASCII alphanumerics, plus any
 * non-CJK codepoint above ASCII so accented Latin, Cyrillic and Greek behave as
 * words rather than as separators. */
static int zkb_is_word(unsigned int cp) {
    if (cp < 0x80) {
        return (cp >= '0' && cp <= '9') || (cp >= 'a' && cp <= 'z') ||
               (cp >= 'A' && cp <= 'Z');
    }
    return !zkb_is_cjk(cp);
}

/* ------------------------------------------------------------ tokenizer ---- */

static int zkb_tok_create(void *pUnused, const char **azArg, int nArg,
                          Fts5Tokenizer **ppOut) {
    (void)pUnused; (void)azArg; (void)nArg;
    /* Stateless: a non-null sentinel is enough, and avoids an allocation per
     * tokenizer instance. */
    *ppOut = (Fts5Tokenizer *)&zkb_tok_create;
    return SQLITE_OK;
}

static void zkb_tok_delete(Fts5Tokenizer *p) { (void)p; }

static int zkb_tok_tokenize(
    Fts5Tokenizer *pTok, void *pCtx, int flags,
    const char *pText, int nText,
    int (*xToken)(void *, int, const char *, int, int, int)
) {
    (void)pTok;
    (void)flags; /* Documents and queries are tokenized identically. */

    const unsigned char *text = (const unsigned char *)pText;
    int i = 0;
    int rc = SQLITE_OK;
    char buf[ZKB_MAX_TOKEN];

    while (i < nText && rc == SQLITE_OK) {
        unsigned int cp;
        int len = zkb_decode(text + i, nText - i, &cp);

        if (zkb_is_cjk(cp)) {
            /* Collect the run, remembering where each character starts so a
             * bigram's byte range can be reported exactly. */
            int starts[512];
            int count = 0;
            int j = i;
            while (j < nText && count < (int)(sizeof(starts) / sizeof(starts[0])) - 1) {
                unsigned int c2;
                int l2 = zkb_decode(text + j, nText - j, &c2);
                if (!zkb_is_cjk(c2)) break;
                starts[count++] = j;
                j += l2;
            }
            starts[count] = j; /* sentinel: end of the run */

            if (count == 1) {
                /* A lone CJK character cannot form a bigram. Emit it as-is so it
                 * is at least findable; single-character queries stay weak, which
                 * is inherent to bigram indexing and reported by the query
                 * builder rather than hidden. */
                rc = xToken(pCtx, 0, (const char *)text + starts[0],
                            starts[1] - starts[0], starts[0], starts[1]);
            } else {
                for (int k = 0; k + 1 < count && rc == SQLITE_OK; k++) {
                    rc = xToken(pCtx, 0, (const char *)text + starts[k],
                                starts[k + 2] - starts[k], starts[k], starts[k + 2]);
                }
            }
            i = j;
            continue;
        }

        if (zkb_is_word(cp)) {
            int start = i;
            int n = 0;
            int j = i;
            while (j < nText) {
                unsigned int c2;
                int l2 = zkb_decode(text + j, nText - j, &c2);
                if (!zkb_is_word(c2)) break;
                if (n + l2 <= ZKB_MAX_TOKEN) {
                    /* ASCII case folding only. Full Unicode folding is out of
                     * scope; unicode61 does not do it either. */
                    for (int b = 0; b < l2; b++) {
                        unsigned char ch = text[j + b];
                        buf[n + b] = (ch >= 'A' && ch <= 'Z') ? (char)(ch + 32) : (char)ch;
                    }
                    n += l2;
                }
                j += l2;
            }
            if (n > 0) rc = xToken(pCtx, 0, buf, n, start, j);
            i = j;
            continue;
        }

        i += len; /* separator */
    }

    return rc;
}

static fts5_tokenizer zkb_cjk_tokenizer = {
    zkb_tok_create,
    zkb_tok_delete,
    zkb_tok_tokenize,
};

/* ---------------------------------------------------------- registration --- */

static int zkb_fts5_api(sqlite3 *db, fts5_api **ppApi) {
    sqlite3_stmt *pStmt = 0;
    *ppApi = 0;
    int rc = sqlite3_prepare_v2(db, "SELECT fts5(?1)", -1, &pStmt, 0);
    if (rc != SQLITE_OK) return rc;
    /* The documented handshake: bind a pointer with the magic type name and
     * fts5 writes its api table into it. */
    sqlite3_bind_pointer(pStmt, 1, (void *)ppApi, "fts5_api_ptr", 0);
    sqlite3_step(pStmt);
    return sqlite3_finalize(pStmt);
}

/* Must be called on every connection, like sqlite3_vec_init: FTS5 tokenizers
 * are per-connection, and a connection without it cannot even open a table that
 * declares tokenize='zkb_cjk'. */
int zkb_register_cjk_tokenizer(sqlite3 *db) {
    fts5_api *pApi = 0;
    int rc = zkb_fts5_api(db, &pApi);
    if (rc != SQLITE_OK) return rc;
    if (pApi == 0) return SQLITE_ERROR;
    return pApi->xCreateTokenizer(pApi, "zkb_cjk", 0, &zkb_cjk_tokenizer, 0);
}
