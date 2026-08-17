/* SQLITE_TRANSIENT is ((sqlite3_destructor_type)-1), a function-pointer cast
 * from -1 that the Zig toolchain rejects. Wrapping the two binds that need it
 * in C is cheaper than fighting the cast. */

#include "sqlite3.h"

int zkb_bind_text_transient(
    sqlite3_stmt *stmt,
    int idx,
    const char *value,
    int n_bytes
) {
    return sqlite3_bind_text(stmt, idx, value, n_bytes, SQLITE_TRANSIENT);
}

int zkb_bind_blob_transient(
    sqlite3_stmt *stmt,
    int idx,
    const void *value,
    int n_bytes
) {
    return sqlite3_bind_blob(stmt, idx, value, n_bytes, SQLITE_TRANSIENT);
}
