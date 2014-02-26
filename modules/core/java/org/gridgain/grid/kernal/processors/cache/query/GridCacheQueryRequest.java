// @java.file.header

/*  _________        _____ __________________        _____
 *  __  ____/___________(_)______  /__  ____/______ ____(_)_______
 *  _  / __  __  ___/__  / _  __  / _  / __  _  __ `/__  / __  __ \
 *  / /_/ /  _  /    _  /  / /_/ /  / /_/ /  / /_/ / _  /  _  / / /
 *  \____/   /_/     /_/   \_,__/   \____/   \__,_/  /_/   /_/ /_/
 */

package org.gridgain.grid.kernal.processors.cache.query;

import org.gridgain.grid.*;
import org.gridgain.grid.cache.*;
import org.gridgain.grid.cache.query.*;
import org.gridgain.grid.kernal.*;
import org.gridgain.grid.kernal.processors.cache.*;
import org.gridgain.grid.lang.*;
import org.gridgain.grid.marshaller.*;
import org.gridgain.grid.util.direct.*;
import org.gridgain.grid.util.typedef.*;
import org.gridgain.grid.util.typedef.internal.*;

import java.io.*;
import java.nio.*;
import java.util.*;

import static org.gridgain.grid.cache.query.GridCacheQueryType.*;

/**
 * Query request.
 *
 * @author @java.author
 * @version @java.version
 */
public class GridCacheQueryRequest<K, V> extends GridCacheMessage<K, V> implements GridCacheDeployable {
    /** */
    private long id;

    /** */
    private String cacheName;

    /** Query id. */
    private int qryId;

    /** */
    private GridCacheQueryType type;

    /** */
    private boolean fields;

    /** */
    private String clause;

    /** */
    private String clsName;

    /** */
    @GridDirectTransient
    private GridPredicate<? super K> keyFilter;

    /** */
    private byte[] keyFilterBytes;

    /** */
    @GridDirectTransient
    private GridPredicate<? super V> valFilter;

    /** */
    private byte[] valFilterBytes;

    /** */
    @GridDirectTransient
    private Runnable beforeCb;

    /** */
    private byte[] beforeCbBytes;

    /** */
    @GridDirectTransient
    private Runnable afterCb;

    /** */
    private byte[] afterCbBytes;

    /** */
    @GridDirectTransient
    private GridPredicate<GridCacheEntry<K, V>> prjFilter;

    /** */
    private byte[] prjFilterBytes;

    /** */
    @GridDirectTransient
    private GridReducer<Map.Entry<K, Object>, Object> rdc;

    /** */
    private byte[] rdcBytes;

    /** */
    @GridDirectTransient
    private GridReducer<List<Object>, Object> fieldsRdc;

    /** */
    private byte[] fieldsRdcBytes;

    /** */
    @GridDirectTransient
    private GridClosure<V, Object> trans;

    /** */
    private byte[] transBytes;

    /** */
    @GridDirectTransient
    private GridPredicate<?> vis;

    /** */
    private byte[] visBytes;

    /** */
    @GridDirectTransient
    private Object[] args;

    /** */
    private byte[] argsBytes;

    /** */
    private byte[] cArgsBytes;

    /** */
    private int pageSize;

    /** */
    private boolean clone;

    /** */
    private boolean incBackups;

    /** */
    private boolean cancel;

    /** */
    private boolean single;

    /** */
    private boolean incMeta;

    /** */
    private boolean all;

    /**
     * Required by {@link Externalizable}
     */
    public GridCacheQueryRequest() {
        // No-op.
    }

    /**
     * @param id Request to cancel.
     * @param fields Fields query flag.
     */
    public GridCacheQueryRequest(long id, boolean fields) {
        this.id = id;
        this.fields = fields;

        cancel = true;
    }

    /**
     * Request to load page.
     *
     * @param id Request ID.
     * @param cacheName Cache name.
     * @param pageSize Page size.
     * @param clone {@code true} if values should be cloned.
     * @param incBackups {@code true} if need to include backups.
     * @param fields Fields query flag.
     * @param all Whether to load all pages.
     */
    public GridCacheQueryRequest(
        long id,
        String cacheName,
        int pageSize,
        boolean clone,
        boolean incBackups,
        boolean fields,
        boolean all) {
        this.id = id;
        this.cacheName = cacheName;
        this.pageSize = pageSize;
        this.clone = clone;
        this.incBackups = incBackups;
        this.fields = fields;
        this.all = all;
    }

    /**
     * @param id Request id.
     * @param cacheName Cache name.
     * @param qryId Query id.
     * @param type Query type.
     * @param fields {@code true} if query returns fields.
     * @param clause Query clause.
     * @param clsName Query class name.
     * @param keyFilter Key filter.
     * @param valFilter Value filter.
     * @param beforeCb Before execution callback.
     * @param afterCb After execution callback.
     * @param prjFilter Projection filter.
     * @param rdc Reducer.
     * @param fieldsRdc Fields query reducer.
     * @param trans Transformer.
     * @param vis Visitor predicate.
     * @param pageSize Page size.
     * @param clone {@code true} if values should be cloned.
     * @param incBackups {@code true} if need to include backups.
     * @param args Query arguments.
     * @param single {@code true} if single result requested, {@code false} if multiple.
     * @param incMeta Include meta data or not.
     */
    public GridCacheQueryRequest(
        long id,
        String cacheName,
        int qryId,
        GridCacheQueryType type,
        boolean fields,
        String clause,
        String clsName,
        GridPredicate<? super K> keyFilter,
        GridPredicate<? super V> valFilter,
        Runnable beforeCb,
        Runnable afterCb,
        GridPredicate<GridCacheEntry<K, V>> prjFilter,
        GridReducer<Map.Entry<K, Object>, Object> rdc,
        GridReducer<List<Object>, Object> fieldsRdc,
        GridClosure<V, Object> trans,
        GridPredicate<?> vis,
        int pageSize,
        boolean clone,
        boolean incBackups,
        Object[] args,
        boolean single,
        boolean incMeta) {
        assert type != null || fields;
        assert clause != null || type == SCAN;
        assert clsName != null || fields || type == SCAN;

        this.id = id;
        this.cacheName = cacheName;
        this.qryId = qryId;
        this.type = type;
        this.fields = fields;
        this.clause = clause;
        this.clsName = clsName;
        this.keyFilter = keyFilter;
        this.valFilter = valFilter;
        this.beforeCb = beforeCb;
        this.afterCb = afterCb;
        this.prjFilter = prjFilter;
        this.rdc = rdc;
        this.fieldsRdc = fieldsRdc;
        this.trans = trans;
        this.vis = vis;
        this.pageSize = pageSize;
        this.clone = clone;
        this.incBackups = incBackups;
        this.args = args;
        this.single = single;
        this.incMeta = incMeta;
    }

    /** {@inheritDoc} */
    @Override public void prepareMarshal(GridCacheContext<K, V> ctx) throws GridException {
        super.prepareMarshal(ctx);

        if (keyFilter != null) {
            if (ctx.deploymentEnabled())
                prepareObject(keyFilter, ctx);

            keyFilterBytes = CU.marshal(ctx, keyFilter);
        }

        if (valFilter != null) {
            if (ctx.deploymentEnabled())
                prepareObject(valFilter, ctx);

            valFilterBytes = CU.marshal(ctx, valFilter);
        }

        if (beforeCb != null) {
            if (ctx.deploymentEnabled())
                prepareObject(beforeCb, ctx);

            beforeCbBytes = CU.marshal(ctx, beforeCb);
        }

        if (afterCb != null) {
            if (ctx.deploymentEnabled())
                prepareObject(afterCb, ctx);

            afterCbBytes = CU.marshal(ctx, afterCb);
        }

        if (prjFilter != null) {
            if (ctx.deploymentEnabled())
                prepareObject(prjFilter, ctx);

            prjFilterBytes = CU.marshal(ctx, prjFilter);
        }

        if (rdc != null) {
            if (ctx.deploymentEnabled())
                prepareObject(rdc, ctx);

            rdcBytes = CU.marshal(ctx, rdc);
        }

        if (fieldsRdc != null) {
            if (ctx.deploymentEnabled())
                prepareObject(fieldsRdc, ctx);

            fieldsRdcBytes = CU.marshal(ctx, fieldsRdc);
        }

        if (trans != null) {
            if (ctx.deploymentEnabled())
                prepareObject(trans, ctx);

            transBytes = CU.marshal(ctx, trans);
        }

        if (vis != null) {
            if (ctx.deploymentEnabled())
                prepareObject(vis, ctx);

            visBytes = CU.marshal(ctx, vis);
        }

        if (!F.isEmpty(args)) {
            if (ctx.deploymentEnabled()) {
                for (Object arg : args)
                    prepareObject(arg, ctx);
            }

            argsBytes = CU.marshal(ctx, args);
        }
    }

    /** {@inheritDoc} */
    @Override public void finishUnmarshal(GridCacheContext<K, V> ctx, ClassLoader ldr) throws GridException {
        super.finishUnmarshal(ctx, ldr);

        GridMarshaller mrsh = ctx.marshaller();

        if (keyFilterBytes != null)
            keyFilter = mrsh.unmarshal(keyFilterBytes, ldr);

        if (valFilterBytes != null)
            valFilter = mrsh.unmarshal(valFilterBytes, ldr);

        if (beforeCbBytes != null)
            beforeCb = mrsh.unmarshal(beforeCbBytes, ldr);

        if (afterCbBytes != null)
            afterCb = mrsh.unmarshal(afterCbBytes, ldr);

        if (prjFilterBytes != null)
            prjFilter = mrsh.unmarshal(prjFilterBytes, ldr);

        if (rdcBytes != null)
            rdc = mrsh.unmarshal(rdcBytes, ldr);

        if (fieldsRdcBytes != null)
            fieldsRdc = mrsh.unmarshal(fieldsRdcBytes, ldr);

        if (transBytes != null)
            trans = mrsh.unmarshal(transBytes, ldr);

        if (visBytes != null)
            vis = mrsh.unmarshal(visBytes, ldr);

        if (argsBytes != null)
            args = mrsh.unmarshal(argsBytes, ldr);
    }

    /**
     * @return Request id.
     */
    public long id() {
        return id;
    }

    /**
     * @return Cache name.
     */
    public String cacheName() {
        return cacheName;
    }

    /**
     * @return Query id.
     */
    public int queryId() {
        return qryId;
    }

    /**
     * @return Query type.
     */
    public GridCacheQueryType type() {
        return type;
    }

    /**
     * @return {@code true} if query returns fields.
     */
    public boolean fields() {
        return fields;
    }

    /**
     * @return Query clause.
     */
    public String clause() {
        return clause;
    }

    /**
     * @return Class name.
     */
    public String className() {
        return clsName;
    }

    /**
     * @return Flag indicating whether to clone values.
     */
    public boolean cloneValues() {
        return clone;
    }

    /**
     * @return Flag indicating whether to include backups.
     */
    public boolean includeBackups() {
        return incBackups;
    }

    /**
     * @return Flag indicating that this is cancel request.
     */
    public boolean cancel() {
        return cancel;
    }

    /**
     * @return Key filter.
     */
    public GridPredicate<? super K> keyFilter() {
        return keyFilter;
    }

    /**
     * @return Value filter.
     */
    public GridPredicate<? super V> valueFilter() {
        return valFilter;
    }

    /**
     * @return Before execution callback.
     */
    public Runnable beforeCallback() {
        return beforeCb;
    }

    /**
     * @return After execution callback.
     */
    public Runnable afterCallback() {
        return afterCb;
    }

    /** {@inheritDoc} */
    public GridPredicate<GridCacheEntry<K, V>> projectionFilter() {
        return prjFilter;
    }

    /**
     * @return Reducer.
     */
    public GridReducer<Map.Entry<K, Object>, Object> reducer() {
        return rdc;
    }

    /**
     * @return Reducer for fields queries.
     */
    public GridReducer<List<Object>, Object> fieldsReducer() {
        return fieldsRdc;
    }

    /**
     * @return Transformer.
     */
    public GridClosure<V, Object> transformer() {
        return trans;
    }

    /**
     * @return Visitor predicate.
     */
    public GridPredicate<?> visitor() {
        return vis;
    }

    /**
     * @return Page size.
     */
    public int pageSize() {
        return pageSize;
    }

    /**
     * @return Arguments.
     */
    public Object[] arguments() {
        return args;
    }

    /**
     * @return {@code true} if single result requested, {@code false} otherwise.
     */
    public boolean single() {
        return single;
    }

    /**
     * @return Include meta data or not.
     */
    public boolean includeMetaData() {
        return incMeta;
    }

    /**
     * @return Whether to load all pages.
     */
    public boolean allPages() {
        return all;
    }

    /** {@inheritDoc} */
    @SuppressWarnings({"CloneDoesntCallSuperClone", "CloneCallsConstructors"})
    @Override public GridTcpCommunicationMessageAdapter clone() {
        GridCacheQueryRequest _clone = new GridCacheQueryRequest();

        clone0(_clone);

        return _clone;
    }

    /** {@inheritDoc} */
    @Override protected void clone0(GridTcpCommunicationMessageAdapter _msg) {
        super.clone0(_msg);

        GridCacheQueryRequest _clone = (GridCacheQueryRequest)_msg;

        _clone.id = id;
        _clone.cacheName = cacheName;
        _clone.qryId = qryId;
        _clone.type = type;
        _clone.fields = fields;
        _clone.clause = clause;
        _clone.clsName = clsName;
        _clone.keyFilter = keyFilter;
        _clone.keyFilterBytes = keyFilterBytes;
        _clone.valFilter = valFilter;
        _clone.valFilterBytes = valFilterBytes;
        _clone.beforeCb = beforeCb;
        _clone.beforeCbBytes = beforeCbBytes;
        _clone.afterCb = afterCb;
        _clone.afterCbBytes = afterCbBytes;
        _clone.prjFilter = prjFilter;
        _clone.prjFilterBytes = prjFilterBytes;
        _clone.rdc = rdc;
        _clone.rdcBytes = rdcBytes;
        _clone.fieldsRdc = fieldsRdc;
        _clone.fieldsRdcBytes = fieldsRdcBytes;
        _clone.trans = trans;
        _clone.transBytes = transBytes;
        _clone.vis = vis;
        _clone.visBytes = visBytes;
        _clone.args = args;
        _clone.argsBytes = argsBytes;
        _clone.cArgsBytes = cArgsBytes;
        _clone.pageSize = pageSize;
        _clone.clone = clone;
        _clone.incBackups = incBackups;
        _clone.cancel = cancel;
        _clone.single = single;
        _clone.incMeta = incMeta;
        _clone.all = all;
    }

    /** {@inheritDoc} */
    @SuppressWarnings("all")
    @Override public boolean writeTo(ByteBuffer buf) {
        commState.setBuffer(buf);

        if (!super.writeTo(buf))
            return false;

        if (!commState.typeWritten) {
            if (!commState.putByte(directType()))
                return false;

            commState.typeWritten = true;
        }

        switch (commState.idx) {
            case 2:
                if (!commState.putByteArray(afterCbBytes))
                    return false;

                commState.idx++;

            case 3:
                if (!commState.putBoolean(all))
                    return false;

                commState.idx++;

            case 4:
                if (!commState.putByteArray(argsBytes))
                    return false;

                commState.idx++;

            case 5:
                if (!commState.putByteArray(beforeCbBytes))
                    return false;

                commState.idx++;

            case 6:
                if (!commState.putByteArray(cArgsBytes))
                    return false;

                commState.idx++;

            case 7:
                if (!commState.putString(cacheName))
                    return false;

                commState.idx++;

            case 8:
                if (!commState.putBoolean(cancel))
                    return false;

                commState.idx++;

            case 9:
                if (!commState.putString(clause))
                    return false;

                commState.idx++;

            case 10:
                if (!commState.putBoolean(clone))
                    return false;

                commState.idx++;

            case 11:
                if (!commState.putString(clsName))
                    return false;

                commState.idx++;

            case 12:
                if (!commState.putBoolean(fields))
                    return false;

                commState.idx++;

            case 13:
                if (!commState.putByteArray(fieldsRdcBytes))
                    return false;

                commState.idx++;

            case 14:
                if (!commState.putLong(id))
                    return false;

                commState.idx++;

            case 15:
                if (!commState.putBoolean(incBackups))
                    return false;

                commState.idx++;

            case 16:
                if (!commState.putBoolean(incMeta))
                    return false;

                commState.idx++;

            case 17:
                if (!commState.putByteArray(keyFilterBytes))
                    return false;

                commState.idx++;

            case 18:
                if (!commState.putInt(pageSize))
                    return false;

                commState.idx++;

            case 19:
                if (!commState.putByteArray(prjFilterBytes))
                    return false;

                commState.idx++;

            case 20:
                if (!commState.putInt(qryId))
                    return false;

                commState.idx++;

            case 21:
                if (!commState.putByteArray(rdcBytes))
                    return false;

                commState.idx++;

            case 22:
                if (!commState.putBoolean(single))
                    return false;

                commState.idx++;

            case 23:
                if (!commState.putByteArray(transBytes))
                    return false;

                commState.idx++;

            case 24:
                if (!commState.putEnum(type))
                    return false;

                commState.idx++;

            case 25:
                if (!commState.putByteArray(valFilterBytes))
                    return false;

                commState.idx++;

            case 26:
                if (!commState.putByteArray(visBytes))
                    return false;

                commState.idx++;

        }

        return true;
    }

    /** {@inheritDoc} */
    @SuppressWarnings("all")
    @Override public boolean readFrom(ByteBuffer buf) {
        commState.setBuffer(buf);

        if (!super.readFrom(buf))
            return false;

        switch (commState.idx) {
            case 2:
                byte[] afterCbBytes0 = commState.getByteArray();

                if (afterCbBytes0 == BYTE_ARR_NOT_READ)
                    return false;

                afterCbBytes = afterCbBytes0;

                commState.idx++;

            case 3:
                if (buf.remaining() < 1)
                    return false;

                all = commState.getBoolean();

                commState.idx++;

            case 4:
                byte[] argsBytes0 = commState.getByteArray();

                if (argsBytes0 == BYTE_ARR_NOT_READ)
                    return false;

                argsBytes = argsBytes0;

                commState.idx++;

            case 5:
                byte[] beforeCbBytes0 = commState.getByteArray();

                if (beforeCbBytes0 == BYTE_ARR_NOT_READ)
                    return false;

                beforeCbBytes = beforeCbBytes0;

                commState.idx++;

            case 6:
                byte[] cArgsBytes0 = commState.getByteArray();

                if (cArgsBytes0 == BYTE_ARR_NOT_READ)
                    return false;

                cArgsBytes = cArgsBytes0;

                commState.idx++;

            case 7:
                String cacheName0 = commState.getString();

                if (cacheName0 == STR_NOT_READ)
                    return false;

                cacheName = cacheName0;

                commState.idx++;

            case 8:
                if (buf.remaining() < 1)
                    return false;

                cancel = commState.getBoolean();

                commState.idx++;

            case 9:
                String clause0 = commState.getString();

                if (clause0 == STR_NOT_READ)
                    return false;

                clause = clause0;

                commState.idx++;

            case 10:
                if (buf.remaining() < 1)
                    return false;

                clone = commState.getBoolean();

                commState.idx++;

            case 11:
                String clsName0 = commState.getString();

                if (clsName0 == STR_NOT_READ)
                    return false;

                clsName = clsName0;

                commState.idx++;

            case 12:
                if (buf.remaining() < 1)
                    return false;

                fields = commState.getBoolean();

                commState.idx++;

            case 13:
                byte[] fieldsRdcBytes0 = commState.getByteArray();

                if (fieldsRdcBytes0 == BYTE_ARR_NOT_READ)
                    return false;

                fieldsRdcBytes = fieldsRdcBytes0;

                commState.idx++;

            case 14:
                if (buf.remaining() < 8)
                    return false;

                id = commState.getLong();

                commState.idx++;

            case 15:
                if (buf.remaining() < 1)
                    return false;

                incBackups = commState.getBoolean();

                commState.idx++;

            case 16:
                if (buf.remaining() < 1)
                    return false;

                incMeta = commState.getBoolean();

                commState.idx++;

            case 17:
                byte[] keyFilterBytes0 = commState.getByteArray();

                if (keyFilterBytes0 == BYTE_ARR_NOT_READ)
                    return false;

                keyFilterBytes = keyFilterBytes0;

                commState.idx++;

            case 18:
                if (buf.remaining() < 4)
                    return false;

                pageSize = commState.getInt();

                commState.idx++;

            case 19:
                byte[] prjFilterBytes0 = commState.getByteArray();

                if (prjFilterBytes0 == BYTE_ARR_NOT_READ)
                    return false;

                prjFilterBytes = prjFilterBytes0;

                commState.idx++;

            case 20:
                if (buf.remaining() < 4)
                    return false;

                qryId = commState.getInt();

                commState.idx++;

            case 21:
                byte[] rdcBytes0 = commState.getByteArray();

                if (rdcBytes0 == BYTE_ARR_NOT_READ)
                    return false;

                rdcBytes = rdcBytes0;

                commState.idx++;

            case 22:
                if (buf.remaining() < 1)
                    return false;

                single = commState.getBoolean();

                commState.idx++;

            case 23:
                byte[] transBytes0 = commState.getByteArray();

                if (transBytes0 == BYTE_ARR_NOT_READ)
                    return false;

                transBytes = transBytes0;

                commState.idx++;

            case 24:
                Object type0 = commState.getEnum(GridCacheQueryType.class);

                if (type0 == ENUM_NOT_READ)
                    return false;

                type = (GridCacheQueryType)type0;

                commState.idx++;

            case 25:
                byte[] valFilterBytes0 = commState.getByteArray();

                if (valFilterBytes0 == BYTE_ARR_NOT_READ)
                    return false;

                valFilterBytes = valFilterBytes0;

                commState.idx++;

            case 26:
                byte[] visBytes0 = commState.getByteArray();

                if (visBytes0 == BYTE_ARR_NOT_READ)
                    return false;

                visBytes = visBytes0;

                commState.idx++;

        }

        return true;
    }

    /** {@inheritDoc} */
    @Override public byte directType() {
        return 57;
    }

    /** {@inheritDoc} */
    @Override public String toString() {
        return S.toString(GridCacheQueryRequest.class, this, super.toString());
    }
}
