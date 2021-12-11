#include "Python.h"
#include <math.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "numpy/ndarraytypes.h"
#include "numpy/ndarrayobject.h"
//#include "numpy/ufuncobject.h"
//#include "numpy/npy_3kcompat.h"
#include <stdio.h>
#include <stdlib.h>


#define MAX(x,y) ((x) > (y) ? (x) : (y))
#define MIN(x,y) ((x) < (y) ? (x) : (y))
#define ABS(x)  ((x) < 0 ? -(x) : (x))

#define PyArray_GETPTR5(obj, i, j, k, l, m) ((void *)(PyArray_BYTES(obj) + \
                                            (i)*PyArray_STRIDES(obj)[0] + \
                                            (j)*PyArray_STRIDES(obj)[1] + \
                                            (k)*PyArray_STRIDES(obj)[2] + \
                                            (l)*PyArray_STRIDES(obj)[3] + \
                                            (m)*PyArray_STRIDES(obj)[4]))
#define PyArray_GETPTR6(obj, i, j, k, l, m, n) ((void *)(PyArray_BYTES(obj) + \
                                            (i)*PyArray_STRIDES(obj)[0] + \
                                            (j)*PyArray_STRIDES(obj)[1] + \
                                            (k)*PyArray_STRIDES(obj)[2] + \
                                            (l)*PyArray_STRIDES(obj)[3] + \
                                            (m)*PyArray_STRIDES(obj)[4] + \
                                            (n)*PyArray_STRIDES(obj)[5]))


#define IN_REQUIREMENTS	(NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_ALIGNED)


static PyObject *
py_convolve(PyObject *dummy, PyObject *args)
{
    /*
    *   Arguments:
    *       a - ndarray, nd=4
    *       f - ndarray, nd=4
    *       mode - int - 0 for "valid" and 1 for "full"
    *
    *    Convolution is done over last 2 dimensions of a and f. f is expected to be smaller
    *    in both of the convolution dimensions.
    *    First, the convolution of the 2 arrays (a1,a2,a3,a4) (f1,f2,f3,f4) produces
    *     (a1,a2,f1,f2,c1,c2)
    *    After the convolution, calculates the "trace" over 2nd dimension of both arrays
    *     (a1,a2,f1,f2,c1,c2) -> (a1,f1,c1,c2)
    */
    //printf("py_convolve\n");

    PyObject *arr1=NULL, *arr2=NULL;
    int mode;
    PyArrayObject *a=NULL, *f=NULL, *z=NULL;
    
    if (!PyArg_ParseTuple(args, "OOi", 
        &arr1, &arr2, 
        &mode)) return NULL;
        
    a = (PyArrayObject *)(PyArray_FROM_OTF(arr1, NPY_DOUBLE, IN_REQUIREMENTS));
    if (a == NULL) goto fail;
    
    f = (PyArrayObject *)(PyArray_FROM_OTF(arr2, NPY_DOUBLE, IN_REQUIREMENTS));
    if (f == NULL) goto fail;
    
    npy_intp *ashape = PyArray_SHAPE(a);
    npy_intp *fshape = PyArray_SHAPE(f);
    
    npy_intp zshape[4];
    
    zshape[0] = ashape[0];
    zshape[1] = fshape[0];

    if( mode == 0 )
    {
        zshape[2] = ashape[2] - fshape[2] + 1;
        zshape[3] = ashape[3] - fshape[3] + 1;
    }
    else
    {
        zshape[2] = ashape[2] + fshape[2] - 1;
        zshape[3] = ashape[3] + fshape[3] - 1;
    }
    
    
    z = (PyArrayObject *) PyArray_SimpleNew(4, zshape, NPY_DOUBLE); /* assume we own the z reference now */
    if( z == NULL ) goto fail;
    
    int xa, ya, xf, yf, xz, yz, i, j, k;
    
    int stride_ay = PyArray_STRIDE(a, 2);
    int stride_ax = PyArray_STRIDE(a, 3);
    
    int stride_fy = PyArray_STRIDE(f, 2);
    int stride_fx = PyArray_STRIDE(f, 3);
    
    if( mode == 0 )
    {
        /* mode is "valid" */
        for(i=0; i<zshape[0]; i++)
		{
            for(j=0; j<zshape[1]; j++)
			{
                for(yz=0; yz<zshape[2]; yz++)
				{
                    for(xz=0; xz<zshape[3]; xz++)
                    {
                        double val = 0;
                        for(k=0; k<ashape[1]; k++)
                        {

                            void *pay = PyArray_GETPTR4(a, i, k, yz, xz);
                            void *pfy = PyArray_GETPTR4(f, j, k, 0, 0);

                            for(yf=0; yf<fshape[2]; yf++)
                            {
                                void *pax = pay;
                                void *pfx = pfy;

                                for(xf=0; xf<fshape[3]; xf++)
                                {
                                    double va = *(double*)pax;
                                    double vf = *(double*)pfx;
                                    //printf("f: %d %d, a: %d %d\n", xf, yf, xa, ya);
                                    val += va * vf;
                                    pax += stride_ax;   
                                    pfx += stride_fx;
                                }
                                
                                pay += stride_ay;      
                                pfy += stride_fy;
                            }
                        }
                        *(double*)PyArray_GETPTR4(z, i, j, yz, xz) = val;
                    }
				}
			}
		}
    }   
    else
    {
        /* mode is "full" with implied 0 padding */
        for(i=0; i<zshape[0]; i++)
            for(j=0; j<zshape[1]; j++)
                for(yz=0; yz<zshape[2]; yz++)
                    for(xz=0; xz<zshape[3]; xz++)
                    {
                        int ya_min = yz-fshape[2]+1;
                        int xa_min = xz-fshape[3]+1;
                        
                        double val = 0;
                        for(k=0; k<ashape[1]; k++)
                        {
                            void *pay = PyArray_GETPTR4(a, i, k, ya_min, xa_min);
                            void *pfy = PyArray_GETPTR4(f, j, k, 0, 0); 

                            for(yf=0, ya = ya_min; yf<fshape[2]; yf++, ya++)
                            {
                                void *pax = pay;
                                void *pfx = pfy;
                                
                                if( ya >=0 && ya < ashape[2] )
                                    for(xf=0, xa=xa_min; xf<fshape[3]; xf++, xa++)
                                    {
                                        if( xa >=0 && xa < ashape[3] )
                                        {
                                            double va = *(double*)pax;
                                            double vf = *(double*)pfx;
                                            val += va * vf;
                                        }
                                        pax += stride_ax;
                                        pfx += stride_fx;
                                    }
                                pay += stride_ay;
                                pfy += stride_fy;
                            }
                        }
                        *(double*)PyArray_GETPTR4(z, i, j, yz, xz) = val;
                    }
    }   
            
        
    Py_DECREF(a);
    Py_DECREF(f);
    /* transfer the ownership of the z reference now */
    return (PyObject *)z;
    
 fail:
    Py_XDECREF(a);
    Py_XDECREF(f);
    Py_XDECREF(z);
    return NULL;
}


static PyObject *
py_pool(PyObject *dummy, PyObject *args)
{
    /*
    *   Arguments:
    *       input - ndarray, nd=3
    *       pool_y  - int
    *       pool_x  - int
    *   Returns tuple:
    *       output - result of pool nd=3 [nb, ny, nx, nc]
    *       pool index - nd = 4:   [nb, ny, nx, nc, 2] nx, ny are dimensions of the output array
    *
    */
    //printf("py_convolve\n");

    PyObject *inp=NULL;
    PyArrayObject *input = NULL, *output = NULL, *pool_index = NULL;
    int pool_x, pool_y;

    if (!PyArg_ParseTuple(args, "Oii", &inp, &pool_y, &pool_x)) return NULL;
    
    int nb, input_x, input_y, input_c;
    int output_x, output_y;
        
    input = (PyArrayObject *)(PyArray_FROM_OTF(inp, NPY_DOUBLE, IN_REQUIREMENTS));
    if (input == NULL)
        return NULL;
    

    npy_intp *in_shape = PyArray_SHAPE(input);
    nb = in_shape[0];
    input_y = in_shape[1];
    input_x = in_shape[2];
    input_c = in_shape[3];
    output_x = (input_x+pool_x-1)/pool_x;
    output_y = (input_y+pool_y-1)/pool_y;
        
    npy_intp out_shape[4];
    
    out_shape[0] = nb;
    out_shape[1] = output_y;
    out_shape[2] = output_x;
    out_shape[3] = input_c;
    
    npy_intp index_shape[5];
    index_shape[0] = nb;
    index_shape[1] = output_y;
    index_shape[2] = output_x;
    index_shape[3] = input_c;
    index_shape[4] = 2;
    
    output = (PyArrayObject *) PyArray_SimpleNew(4, out_shape, NPY_DOUBLE); 
    if( output == NULL )
    {
        Py_XDECREF(input);
        return NULL;
    }
    
    pool_index = (PyArrayObject *) PyArray_SimpleNew(5, index_shape, NPY_INT); 
    if( pool_index == NULL )
    {
        Py_XDECREF(input);
        Py_XDECREF(output);
        return NULL;
    }     
    
    int ib, x, y, c, dx, dy, dxmax, dymax;
    double vmax;
    
    int stride_y = PyArray_STRIDE(input, 1);
    int stride_x = PyArray_STRIDE(input, 2);
    
    int stride_inx = PyArray_STRIDE(pool_index, 4);

    //printf("loop...\n");

    for( ib=0; ib < nb; ib++)
        for( y=0; y<output_y; y++ )
            for( x=0; x<output_x; x++ )
                for( c=0; c<input_c; c++ )
                {
                    int y0 = y*pool_y;
                    int x0 = x*pool_x;
                    void *py = PyArray_GETPTR4(input, ib, y0, x0, c);
                    vmax = *(double*)py;
                    dxmax = dymax = 0;
                    for( dy = 0; dy < pool_y && y0 + dy < input_y; dy++ )
                    {
                        void *px = py;
                        for( dx = 0; dx < pool_x && x0 + dx < input_x; dx++ )
                        {
                            double v = *(double*)px;
                            //double v = *(double*)PyArray_GETPTR3(input, yy, xx, c);
                            if( v > vmax )
                            {
                                vmax = v;
                                dxmax = dx;
                                dymax = dy;
                            }
                            px += stride_x;
                        }
                        py += stride_y;
                    }
                    //printf("%d, %d, %d: vmax=%f at %d %d\n", y, x, c, vmax, dymax, dxmax);
                    *(double*)PyArray_GETPTR4(output, ib, y, x, c) = vmax;
                    
                    npy_intp index_inx[5] = {ib, y, x, c, 0};
                    void *index_ptr = PyArray_GetPtr(pool_index, index_inx);
                    *(int*)index_ptr = dymax;
                    *(int*)(index_ptr+stride_inx) = dxmax;
                }
    //printf( "Building tuple...\n");
    PyObject *out = Py_BuildValue("(OO)", (PyObject*)output, (PyObject*)pool_index);
    Py_DECREF(input);
    Py_DECREF(output);
    Py_DECREF(pool_index);
    //printf("returning\n");
    return out;
}
    

static PyObject *
py_pool_back(PyObject *dummy, PyObject *args)
{
    /*
    *   Arguments:
    *       gradient on the output side - ndarray, nd=3 [nb, ny, nx, nc]
    *       pool_index  [nb, ny, nx, nc, 2]
    *       pool_y  - int
    *       pool_x  - int
    *       out_y   - int
    *       out_x   - int
    */

    PyObject *arr1=NULL, *arr2=NULL;
    PyArrayObject *grads = NULL, *output = NULL, *pool_index = NULL;
    int pool_x, pool_y, output_x, output_y;

    if (!PyArg_ParseTuple(args, "OOiiii", &arr1, &arr2, 
        &pool_y, &pool_x, &output_y, &output_x)) return NULL;
    
    grads = (PyArrayObject *)(PyArray_FROM_OTF(arr1, NPY_DOUBLE, IN_REQUIREMENTS));
    if (grads == NULL)
    {
        return NULL;
    }

    pool_index = (PyArrayObject *)(PyArray_FROM_OTF(arr2, NPY_INT, IN_REQUIREMENTS));
    if (pool_index == NULL)
    {
        Py_DECREF(grads);
        return NULL;
    }

    int grad_x, grad_y;
    int nb, channels, dtype;
    int index_x, index_y;
        
    npy_intp *grad_shape = PyArray_SHAPE(grads);
    nb = grad_shape[0];
    grad_y = grad_shape[1];
    grad_x = grad_shape[2];
    channels = grad_shape[3];
    dtype = PyArray_TYPE(grads);
    
    npy_intp *index_shape = PyArray_SHAPE(pool_index);
    index_y = index_shape[1];
    index_x = index_shape[2];
    
    npy_intp out_shape[4];
    out_shape[0] = nb;
    out_shape[1] = output_y;
    out_shape[2] = output_x;
    out_shape[3] = channels;
        
    output = (PyArrayObject *)PyArray_ZEROS(4, out_shape, NPY_DOUBLE, 0);
    if( output == NULL )
    {
        Py_DECREF(grads);
        Py_DECREF(pool_index);
        return NULL;
    }
    
    //printf("output created: %d %d %d %d\n", nb, output_y, output_x, channels);
    
    //return (PyObject*)output;
    
    int grad_stride_b = PyArray_STRIDE(grads, 0);
    int grad_stride_y = PyArray_STRIDE(grads, 1);
    int grad_stride_x = PyArray_STRIDE(grads, 2);
    int grad_stride_c = PyArray_STRIDE(grads, 3);
    
    int index_stride_b = PyArray_STRIDE(pool_index, 0);
    int index_stride_y = PyArray_STRIDE(pool_index, 1);
    int index_stride_x = PyArray_STRIDE(pool_index, 2);
    int index_stride_c = PyArray_STRIDE(pool_index, 3);
    int index_stride_d = PyArray_STRIDE(pool_index, 4);

    void *gpb = PyArray_GETPTR4(grads, 0, 0, 0, 0);
    
    //printf("grad[0,0,0]=%f\n", *(double*)gpy);
    
    npy_intp zeros[5] = {0,0,0,0,0};
    void *ipb = PyArray_GetPtr(pool_index, zeros);

    int ib, x, y, c;
    for( ib=0; ib<nb; ib++ )
    {
        void *gpy = gpb;
        void *ipy = ipb;
        
        for( y=0; y<index_y; y++ )
        {
            void *gpx = gpy;
            void *ipx = ipy;
            for( x=0; x<index_x; x++ )
            {
                void *gpc = gpx;
                void *ipc = ipx;
                for( c=0; c<channels; c++ )
                {
                    int dy = *(int*)ipc;
                    int dx = *(int*)(ipc + index_stride_d);
                
#if 0
                    if( 0 )
                        printf("%d, %d, %d, %d, %d+%d, %d+%d <- %f\n", 
                            ib, y, x, c,
                                y*pool_y, dy, x*pool_x, dx, 
                                *(double*)gpc);
#endif
                    *(double*)PyArray_GETPTR4(output, ib, y*pool_y+dy, x*pool_x+dx, c) = 
                        //*(double*)PyArray_GETPTR4(grads, ib, y, x, c);
                        *(double*)gpc;
                    gpc += grad_stride_c;
                    ipc += index_stride_c;
                }
                gpx += grad_stride_x;
                ipx += index_stride_x;
            }
            gpy += grad_stride_y;
            ipy += index_stride_y;
        }
        gpb += grad_stride_b;
        ipb += index_stride_b;
    }
    Py_DECREF(grads);
    Py_DECREF(pool_index);
    return (PyObject*)output;
}
    
//extern PyObject *py_convolve_3d();
//extern PyObject *py_convolve_back_3d();
//extern PyObject *py_pool_3d();
//extern PyObject *py_pool_back_3d();



PyObject *py_convolve_3d(PyObject *dummy, PyObject *args)
{
    /*
    *   Arguments:
    *       a - ndarray, nd=5   [n, nax, nay, naz, ci]
    *       f - ndarray, nd=5   [co, nfx, nfy, nfz, ci]
    *       s1 - int  
    *       s2 - int
    *       nout - tuple
    *
    *    Convolution is done over last nx, ny, nz dimenstions, and then summation is over ci.
    *
    *    Output shape is [n, NX, NY, NZ, co]
    *
    *    s1 and s2 are strides. One of them must be = 1. They are same in each direction, x,y,z.
    *    Convolution is each dimenstion is computed as c_i = sum_k ( f_k * a_(s1*i+s2*k) )
    *
    *    nout - shape of the output array (nx, ny, nz), xyz only, used only when s2 != 1
    */

    PyObject *arr1=NULL, *arr2=NULL;
    PyArrayObject *a=NULL, *f=NULL, *z=NULL;
    
    int s1 = 1, s2 = 1;
    long nout_x = 0, nout_y = 0, nout_z = 0;
    
    if (!PyArg_ParseTuple(args, "OOii|(iii)", &arr1, &arr2, &s1, &s2, &nout_x, &nout_y, &nout_z)) return NULL;
    //if (!PyArg_ParseTuple(args, "OO", &arr1, &arr2)) return NULL;
    
    a = (PyArrayObject*)arr1;
    if (a == NULL) goto fail;
    npy_intp *ashape = PyArray_SHAPE(a);
    //printf("a shape:%d,%d,%d,%d,%d\n", ashape[0], ashape[1], ashape[2], ashape[3], ashape[4]);
    
    
    //printf("args:%d,%d,%d,%d,%d\n", s1, s2, nout_x, nout_y, nout_z);
    //printf("arr1=%x\n", arr1);
    //a = (PyArrayObject *)(PyArray_FROM_OTF(arr1, NPY_DOUBLE, IN_REQUIREMENTS));
    //printf("a converted\n");
    
    //printf("converting f\n");
    
    f = (PyArrayObject *)arr2;           //(PyArray_FROM_OTF(arr2, NPY_DOUBLE, IN_REQUIREMENTS));
    //printf("f converted\n");
    if (f == NULL) goto fail;
    

    npy_intp *fshape = PyArray_SHAPE(f);
    
    //printf("fshape: %d,%d,%d,%d,%d\n", fshape[0], fshape[1], fshape[2], fshape[3], fshape[4]);
    
    npy_intp zshape[5];
    int nb = ashape[0];
    int nci = ashape[4];
    int nco = fshape[0];

    
    zshape[0] = nb;
    zshape[4] = nco;

    if( s2 > 1 )
    {
        zshape[1] = nout_x;
        zshape[2] = nout_y;
        zshape[3] = nout_z;
        s1 = 1;
    }
    else
    {   // s2 = 1, s1 >= 1
        zshape[1] = (ashape[1] - fshape[1] + s1)/s1;
        zshape[2] = (ashape[2] - fshape[2] + s1)/s1;
        zshape[3] = (ashape[3] - fshape[3] + s1)/s1;   
        s2 = 1;     
    }
    
    //printf("zshape=(%d,%d,%d,%d,%d)\n", zshape[0], zshape[1], zshape[2], zshape[3],zshape[4]);
    
    z = (PyArrayObject *) PyArray_SimpleNew(5, zshape, NPY_DOUBLE); /* assume we own the z reference now */
    if( z == NULL ) goto fail;
    
    //printf("z allocated\n");
    
    int xf, yf, zf, xz, yz, zz, n, co;
    
    int stride_ax = PyArray_STRIDE(a, 1) * s2;
    int stride_ay = PyArray_STRIDE(a, 2) * s2;
    int stride_az = PyArray_STRIDE(a, 3) * s2;
    int stride_ac = PyArray_STRIDE(a, 4);
    
    int stride_fx = PyArray_STRIDE(f, 1);
    int stride_fy = PyArray_STRIDE(f, 2);
    int stride_fz = PyArray_STRIDE(f, 3);
    int stride_fc = PyArray_STRIDE(f, 4);
    
#if 0
    int xa, ya, za;
    for(n=0; n<nb; n++)
        for(co=0; co<nco; co++)
            for(xa=0; xa<ashape[1]; xa++)
                for(ya=0; ya<ashape[2]; ya++)
                    for(za=0; za<ashape[3]; za++)
                    {   
                        double v = *(double*)PyArray_GETPTR5(a, n, xa, ya, za, co);
                        if( ABS(v) > 1.0e10 )
                            printf("%d, %d, %d, %d, %d -> %g\n", n, xa, ya, za, co, v);
                    }
#endif
                
    
    
    for(n=0; n<nb; n++)
        for(co=0; co<nco; co++)
            for(xz=0; xz<zshape[1]; xz++)
                for(yz=0; yz<zshape[2]; yz++)
                    for(zz=0; zz<zshape[3]; zz++)
                    {
                        //printf("n,co,x,y,z=%d,%d,%d,%d,%d\n", n,co,xz,yz,zz);
                        
                        double val = 0;
                        //int first = 1;

                        void *pax = PyArray_GETPTR5(a, n, xz*s1, yz*s1, zz*s1, 0);
                        void *pfx = PyArray_GETPTR5(f, co, 0, 0, 0, 0);

                        for(xf=0; xf<fshape[1]; xf++)
                        {
                            void *pay = pax;
                            void *pfy = pfx;

                            for(yf=0; yf<fshape[2]; yf++)
                            {
                                void *paz = pay;
                                void *pfz = pfy;

                                for(zf=0; zf<fshape[3]; zf++)
                                {
                                    void *pac = paz;
                                    void *pfc = pfz;
                                    int ci;
                                    for(ci=0; ci<nci; ci++)
                                    {
                                        double va = *(double*)pac;
                                        double vf = *(double*)pfc;
                                        
                                        //va = *(double*)PyArray_GETPTR5(a, n, xz*s1+xf*s2, yz*s1+yf*s2, zz*s1+zf*s2, ci);
                                        
                                        //if( (xz == zshape[1]-1 && yz == zshape[2]-1 && zz == zshape[3]-1 && co == 0 ) )
                                        //    printf("%d, %d, %d, %d: va=%g vf=%g\n", xz, yz, zz, co, va, vf);
                                        
                                
                                        val += va * vf;
                                        pac += stride_ac;
                                        pfc += stride_fc;
#if 0
                                        if( first && ABS(val) > 1000.0 )
                                        {
                                            printf("n, iax,iay,iaz, ci:%d, %d,%d,%d, %d va=%g\n", 
                                                n, xz*s1+xf*s2, yz*s1+yf*s2, zz*s1+zf*s2, ci, va);
                                            first = 0;
                                        }
#endif
                                        
                                    }
                                    paz += stride_az;   
                                    pfz += stride_fz;
                                }
                                pay += stride_ay;   
                                pfy += stride_fy;
                            }
                            pax += stride_ax;      
                            pfx += stride_fx;
                        }
                        *(double*)PyArray_GETPTR5(z, n, xz, yz, zz, co) = val;
                    }
        
    //Py_DECREF(a);
    //Py_DECREF(f);
    /* transfer the ownership of the z reference now */
    return (PyObject *)z;
    
 fail:
    printf("fail");
    //Py_XDECREF(a);
    //Py_XDECREF(f);
    Py_XDECREF(z);
    return NULL;
}

PyObject *py_convolve_back_3d(PyObject *dummy, PyObject *args)
{
    /*
    *   Arguments:
    *       a - ndarray, nd=5   [n, nax, nay, naz, ci]
    *       f - ndarray, nd=5   [co, nfx, nfy, nfz, ci]
    *       s - stride, int
    *       nout - output array shape, tuple of ints: (nout_x, nout_y nout_z)
    *
    *    Convolution is done over last nx, ny, nz dimenstions, and then summation is over ci.
    *
    *    Reversed, zero-padded convoilution. In each direction, convolution is calcilated as:
    *    c_i = sum_k ( a_k * f_(i-s*k) )   where s is the stride
    *
    *    Output shape is [n, nout_x, nout_y nout_z, co]
    *
    */
    //printf("py_convolve\n");

    PyObject *arr1=NULL, *arr2=NULL;
    int s;
    int nout_x = -1, nout_y = -1, nout_z = -1;
    PyArrayObject *a=NULL, *f=NULL, *z=NULL;
    
    if (!PyArg_ParseTuple(args, "OOi(iii)", 
        &arr1, &arr2, &s, &nout_x, &nout_y, &nout_z)) return NULL;
        
    a = (PyArrayObject *)arr1;  //(PyArray_FROM_OTF(arr1, NPY_DOUBLE, IN_REQUIREMENTS));
    if (a == NULL) goto fail;
    
    f = (PyArrayObject *)arr2;  //(PyArray_FROM_OTF(arr2, NPY_DOUBLE, IN_REQUIREMENTS));
    if (f == NULL) goto fail;
    
    npy_intp *ashape = PyArray_SHAPE(a);
    npy_intp *fshape = PyArray_SHAPE(f);
    
    npy_intp zshape[5];
    
    int nb = ashape[0];
    int nci = ashape[4];
    int nco = fshape[0];
    
    zshape[0] = nb;
    zshape[4] = nco;

    zshape[1] = nout_x;
    zshape[2] = nout_y;
    zshape[3] = nout_z;
    
    z = (PyArrayObject *) PyArray_SimpleNew(5, zshape, NPY_DOUBLE); /* assume we own the z reference now */
    if( z == NULL ) goto fail;
    
    int ix, iy, iz, n, co;
    
    int stride_ax = PyArray_STRIDE(a, 1);
    int stride_ay = PyArray_STRIDE(a, 2);
    int stride_az = PyArray_STRIDE(a, 3);
    int stride_ac = PyArray_STRIDE(a, 4);
    
    int stride_fx = PyArray_STRIDE(f, 1)*s;
    int stride_fy = PyArray_STRIDE(f, 2)*s;
    int stride_fz = PyArray_STRIDE(f, 3)*s;
    int stride_fc = PyArray_STRIDE(f, 4);
    
    for(n=0; n<nb; n++)
        for(co=0; co<nco; co++)
            for(ix=0; ix<zshape[1]; ix++)
            {
                int kx_min = MAX(0, (ix-fshape[1]+s)/s);
                int kx_max = MIN(ashape[1], ix/s+1);
                
                for(iy=0; iy<zshape[2]; iy++)
                {
                    int ky_min = MAX(0, (iy-fshape[2]+s)/s);
                    int ky_max = MIN(ashape[2], iy/s+1);

                    for(iz=0; iz<zshape[3]; iz++)
                    {
                        int kz_min = MAX(0, (iz-fshape[3]+s)/s);
                        int kz_max = MIN(ashape[3], iz/s+1);
                    
                        double val = 0;
                        
                        int kx;
                        
                        void *pax = PyArray_GETPTR5(a, n, kx_min, ky_min, kz_min, 0);
                        void *pfx = PyArray_GETPTR5(f, co, ix-s*kx_min, iy-s*ky_min, iz-s*kz_min, 0);

                        for(kx = kx_min; kx < kx_max; kx++)
                        {
                            int ky;
                            void *pay = pax;
                            void *pfy = pfx;
                            for(ky = ky_min; ky < ky_max; ky++)
                            {
                                int kz;
                                void *paz = pay;
                                void *pfz = pfy;
                                for(kz = kz_min; kz < kz_max; kz++)
                                {
                                    int ci;
                                    void *pac = paz;
                                    void *pfc = pfz;
                                    for(ci=0; ci<nci; ci++)
                                    {
                                        //double va = *(double*)PyArray_GETPTR5(a, n, kx, ky, kz, ci);
                                        //double vf = *(double*)PyArray_GETPTR5(f, co,  
                                        //        ix-s*kx, iy-s*ky, iz-s*kz, ci);
                                        double va = *(double*)pac;
                                        double vf = *(double*)pfc;
                                        val += va * vf;
                                        pac += stride_ac;
                                        pfc += stride_fc;
                                    }
                                    paz += stride_az;
                                    pfz -= stride_fz;
                                }
                                pay += stride_ay;
                                pfy -= stride_fy;
                            }
                            pax += stride_ax;
                            pfx -= stride_fx;
                        }
                        *(double*)PyArray_GETPTR5(z, n, ix, iy, iz, co) = val;
                    }
                }
            }
            
        
    //Py_DECREF(a);
    //Py_DECREF(f);
    /* transfer the ownership of the z reference now */
    return (PyObject *)z;
    
 fail:
    //Py_XDECREF(a);
    //Py_XDECREF(f);
    Py_XDECREF(z);
    return NULL;
}



PyObject *py_pool_3d(PyObject *dummy, PyObject *args)
{
    /*
    *   Arguments:
    *       input - ndarray, nd=4
    *       pool_x  - int
    *       pool_x  - int
    *       pool_z  - int
    *   Returns tuple:
    *       output - result of pool nd=4 [nb, nx, ny, nz, nc]
    *       pool index - nd = 5:   [nb, nx, ny, nz, nc, 3] nx, ny, nz are dimensions of the output array
    *
    */
    //printf("py_convolve\n");

    PyObject *inp=NULL;
    PyArrayObject *input = NULL, *output = NULL, *pool_index = NULL;
    int pool_x, pool_y, pool_z;

    if (!PyArg_ParseTuple(args, "Oiii", &inp, &pool_x, &pool_y, &pool_z)) return NULL;
    
    int nb, input_x, input_y, input_z, input_c;
    int output_x, output_y, output_z;
        
    input = (PyArrayObject *)inp;   //(PyArray_FROM_OTF(inp, NPY_DOUBLE, IN_REQUIREMENTS));
    if (input == NULL)
        return NULL;
    

    npy_intp *in_shape = PyArray_SHAPE(input);
    nb =        in_shape[0];
    input_x =   in_shape[1];
    input_y =   in_shape[2];
    input_z =   in_shape[3];
    input_c =   in_shape[4];
    output_x =  (input_x+pool_x-1)/pool_x;
    output_y =  (input_y+pool_y-1)/pool_y;
    output_z =  (input_z+pool_z-1)/pool_z;
	
	//printf("pool3d: output shape: %d %d %d\n", output_z, output_y, output_z );
        
    npy_intp out_shape[5];
    
    out_shape[0] = nb;
    out_shape[1] = output_x;
    out_shape[2] = output_y;
    out_shape[3] = output_z;
    out_shape[4] = input_c;
    
    npy_intp index_shape[6];
    index_shape[0] = nb;
    index_shape[1] = output_x;
    index_shape[2] = output_y;
    index_shape[3] = output_z;
    index_shape[4] = input_c;
    index_shape[5] = 3;
    
    output = (PyArrayObject *) PyArray_SimpleNew(5, out_shape, NPY_DOUBLE); 
    if( output == NULL )
    {
        Py_XDECREF(input);
        return NULL;
    }
    
    pool_index = (PyArrayObject *) PyArray_SimpleNew(6, index_shape, NPY_INT); 
    if( pool_index == NULL )
    {
        Py_XDECREF(input);
        Py_XDECREF(output);
        return NULL;
    }     
    
    int ib, x, y, z, c, dx, dy, dz;
    double vmax;
    
    int stride_x = PyArray_STRIDE(input, 1);
    int stride_y = PyArray_STRIDE(input, 2);
    int stride_z = PyArray_STRIDE(input, 3);
    
    int stride_inx = PyArray_STRIDE(pool_index, 5);

    //printf("loop...\n");

    for( ib=0; ib < nb; ib++)
        for( x=0; x<output_x; x++ )
        {
            int x0 = x*pool_x;
            for( y=0; y<output_y; y++ )
            {
                int y0 = y*pool_y;
                for( z=0; z<output_z; z++ )
                {
                    int z0 = z*pool_z;
                    for( c=0; c<input_c; c++ )
                    {
                        void *px = PyArray_GETPTR5(input, ib, x0, y0, z0, c);
                        vmax = *(double*)px;
                        int dxmax = 0, dymax = 0, dzmax = 0;
                        for( dx = 0; dx < pool_x && x0 + dx < input_x; dx++ )
                        {
                            void *py = px;
                            for( dy = 0; dy < pool_y && y0 + dy < input_y; dy++ )
                            {
                                void *pz = py;
                                for( dz = 0; dz < pool_z && z0 + dz < input_z; dz++ )
                                {
                                    //double v = *(double*)pz;
                                    //void *pxyz = PyArray_GETPTR5(input, ib, x0+dx, y0+dy, z0+dz, c);
                                    double v = *(double*)pz;
                                    //printf("ib/x/y/z/c=%d,%d,%d,%d,%d dx/dy/dz=%d,%d,%d dp=%d/%d %s\n", 
                                    //    ib, x, y, z, c,
                                    //    dx, dy, dz, 
                                    //    pz-px, pxyz - px, pz==pxyz ? "" : "<---");
                                    if( v > vmax )
                                    {
                                        vmax = v;
                                        dxmax = dx;
                                        dymax = dy;
                                        dzmax = dz;
                                    }
                                    pz += stride_z;
                                }
                                py += stride_y;
                            }
                            px += stride_x;
                        }
                        //printf("%d, %d, %d, %d: vmax=%f at %d %d %d\n", y, x, z, c, vmax, dymax, dxmax, dzmax);
                        *(double*)PyArray_GETPTR5(output, ib, x, y, z, c) = vmax;
						//printf("vmax stored\n");
                        void *p = PyArray_GETPTR6(pool_index, ib, x, y, z, c, 0);
                        *(int*)p = dxmax;
						//printf("dx stored\n");
                        *(int*)(p+stride_inx) = dymax;
						//printf("dy stored\n");
                        *(int*)(p+stride_inx*2) = dzmax;
						//printf("dz stored\n");
                    }
                }
            }
        }
	//printf( "Building tuple...\n");
    PyObject *out = Py_BuildValue("(OO)", (PyObject*)output, (PyObject*)pool_index);
    //Py_DECREF(input);
    Py_DECREF(output);
    Py_DECREF(pool_index);
    //printf("returning\n");
    return out;
    
}
    
PyObject *py_pool_back_3d(PyObject *dummy, PyObject *args)
{
    /*
    *   Arguments:
    *       gradient on the output side - ndarray, nd=4 [nb, nx, ny, nz, nc]
    *       pool_index  [nb, nx, ny, nz, nc, 3]
    *       pool_x  - int
    *       pool_y  - int
    *       pool_z  - int
    *       out_x   - int
    *       out_y   - int
    *       out_z   - int
    */

    PyObject *arr1=NULL, *arr2=NULL;
    PyArrayObject *grads = NULL, *output = NULL, *pool_index = NULL;
    int pool_x, pool_y, pool_z, output_x, output_y, output_z;

    if (!PyArg_ParseTuple(args, "OOiiiiii", &arr1, &arr2, 
        &pool_x, &pool_y, &pool_z, &output_x, &output_y, &output_z)) return NULL;
    
    grads = (PyArrayObject *)arr1;  //(PyArray_FROM_OTF(arr1, NPY_DOUBLE, IN_REQUIREMENTS));
    if (grads == NULL)
    {
        return NULL;
    }

    pool_index = (PyArrayObject *)arr2; //(PyArray_FROM_OTF(arr2, NPY_INT, IN_REQUIREMENTS));
    if (pool_index == NULL)
    {
        //Py_DECREF(grads);
        return NULL;
    }

    int grad_x, grad_y, grad_z;
    int nb, channels, dtype;
    int index_x, index_y, index_z;
        
    npy_intp *grad_shape = PyArray_SHAPE(grads);
    nb      = grad_shape[0];
    grad_x  = grad_shape[1];
    grad_y  = grad_shape[2];
    grad_z  = grad_shape[3];
    channels = grad_shape[4];
    dtype = PyArray_TYPE(grads);
    
    npy_intp *index_shape = PyArray_SHAPE(pool_index);
    index_x = index_shape[1];
    index_y = index_shape[2];
    index_z = index_shape[3];
    
    npy_intp out_shape[5];
    out_shape[0] = nb;
    out_shape[1] = output_x;
    out_shape[2] = output_y;
    out_shape[3] = output_z;
    out_shape[4] = channels;
        
    output = (PyArrayObject *)PyArray_ZEROS(5, out_shape, NPY_DOUBLE, 0);
    if( output == NULL )
    {
        //Py_DECREF(grads);
        //Py_DECREF(pool_index);
        return NULL;
    }
    
    //printf("output created: %d %d %d %d\n", nb, output_y, output_x, channels);
    
    //return (PyObject*)output;
    
    int grad_stride_b = PyArray_STRIDE(grads, 0);
    int grad_stride_x = PyArray_STRIDE(grads, 1);
    int grad_stride_y = PyArray_STRIDE(grads, 2);
    int grad_stride_z = PyArray_STRIDE(grads, 3);
    int grad_stride_c = PyArray_STRIDE(grads, 4);
    
    int index_stride_b = PyArray_STRIDE(pool_index, 0);
    int index_stride_x = PyArray_STRIDE(pool_index, 1);
    int index_stride_y = PyArray_STRIDE(pool_index, 2);
    int index_stride_z = PyArray_STRIDE(pool_index, 3);
    int index_stride_c = PyArray_STRIDE(pool_index, 4);
    int index_stride_d = PyArray_STRIDE(pool_index, 5);

    void *gpb = PyArray_GETPTR5(grads, 0, 0, 0, 0, 0);
    void *ipb = PyArray_GETPTR6(pool_index, 0,0,0,0,0,0);

    int ib, x, y, z, c;
    int x0, y0, z0;
    for( ib=0; ib<nb; ib++ )
    {
        void *gpx = gpb;
        void *ipx = ipb;
        for( x = x0 = 0; x<index_x; x++, x0 += pool_x)
        {
            void *gpy = gpx;
            void *ipy = ipx;
            for( y = y0 = 0; y<index_y; y++, y0 += pool_y )
            {
                void *gpz = gpy;
                void *ipz = ipy;
                for( z = z0 = 0; z<index_z; z++, z0 += pool_z )
                {
                    void *gpc = gpz;
                    void *ipc = ipz;
                    for( c=0; c<channels; c++ )
                    {
                        int dx = *(int*)ipc;
                        int dy = *(int*)(ipc + index_stride_d);
                        int dz = *(int*)(ipc + index_stride_d*2);
                
                        *(double*)PyArray_GETPTR5(output, ib, x0+dx, y0+dy, z0+dz, c) = *(double*)gpc;
                        gpc += grad_stride_c;
                        ipc += index_stride_c;
                    }
                    gpz += grad_stride_z;
                    ipz += index_stride_z;
                }
                gpy += grad_stride_y;
                ipy += index_stride_y;
            }
            gpx += grad_stride_x;
            ipx += index_stride_x;
        }
        gpb += grad_stride_b;
        ipb += index_stride_b;
    }
    //Py_DECREF(grads);
    //Py_DECREF(pool_index);
    return (PyObject*)output;
}
    



static PyMethodDef module_methods[] = {
    {"convolve", (PyCFunction) py_convolve, METH_VARARGS, "Convolution 2D"},
    {"pool", (PyCFunction) py_pool, METH_VARARGS, "Pool(max) 2D"},
    {"pool_back", (PyCFunction) py_pool_back, METH_VARARGS, "Gradient backpropagation for pool, 2D"},
    {"convolve_3d", (PyCFunction) py_convolve_3d, METH_VARARGS, "Convolution 3D"},
    {"convolve_back_3d", (PyCFunction) py_convolve_back_3d, METH_VARARGS, "Convolution 3D"},
    {"pool_3d", (PyCFunction) py_pool_3d, METH_VARARGS, "Pool(max) 3D"},
    {"pool_back_3d", (PyCFunction) py_pool_back_3d, METH_VARARGS, "Gradient backpropagation for pool, 3D"},
    {NULL}  /* Sentinel */
};

    
    
#if 0
#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
#endif


static struct PyModuleDef cconvmodule = {
    PyModuleDef_HEAD_INIT,
    "cconv",   /* name of module */
    "Low level convolution network library", /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    module_methods
};


PyMODINIT_FUNC
PyInit_cconv(void) 
{
    PyObject* m;

    m = PyModule_Create(&cconvmodule);
    import_array();
	return m;
}
