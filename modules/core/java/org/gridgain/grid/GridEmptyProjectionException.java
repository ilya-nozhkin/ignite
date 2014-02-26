// @java.file.header

/*  _________        _____ __________________        _____
 *  __  ____/___________(_)______  /__  ____/______ ____(_)_______
 *  _  / __  __  ___/__  / _  __  / _  / __  _  __ `/__  / __  __ \
 *  / /_/ /  _  /    _  /  / /_/ /  / /_/ /  / /_/ / _  /  _  / / /
 *  \____/   /_/     /_/   \_,__/   \____/   \__,_/  /_/   /_/ /_/
 */

package org.gridgain.grid;

import org.jetbrains.annotations.*;

/**
 * This exception defines illegal call on empty projection. Thrown by projection when operation
 * that requires at least one node is called on empty projection.
 *
 * @author @java.author
 * @version @java.version
 */
public class GridEmptyProjectionException extends GridException {
    /**
     * Creates new exception with default error message.
     */
    public GridEmptyProjectionException() {
        super("Grid projection is empty.");
    }

    /**
     * Creates new exception with given error message.
     *
     * @param msg Error message.
     */
    public GridEmptyProjectionException(String msg) {
        super(msg);
    }

    /**
     * Creates a new exception with given error message and optional nested cause exception.
     *
     * @param msg Error message.
     * @param cause Optional nested exception (can be {@code null}).
     */
    public GridEmptyProjectionException(String msg, @Nullable Throwable cause) {
        super(msg, cause);
    }
}
