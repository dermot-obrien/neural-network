/*
 * The MIT License
 *
 * Copyright 2018 Ahmed Tarek.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
package org.fjnn.activation;

import java.nio.FloatBuffer;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.cuda.CudaModule;
import org.fjnn.cuda.CUdeviceptr2D;
import org.fjnn.cuda.CudaFunctions;
import org.fjnn.cuda.CudaUtil;
import org.fjnn.util.intrinsic;

/**
 *
 * @author ahmed
 */
public class LeakyReLU extends Activation {
    public static final float DEFAULT_ALPHA = 0.3f;
    
    final float alpha;
    
    public LeakyReLU() {
        this.alpha = DEFAULT_ALPHA;
    }
    
    public LeakyReLU(float alpha) {
        this.alpha = alpha;
    }
    
    @Override
    public float compute(float input) {
        return input < 0 ? input * alpha : input;
    }
    
    @Override
    public void compute(float[] input, int stride, int count) {
        for(int i=0; i < input.length; i++)
            if(input[i] < 0)
                input[i] = input[i] * alpha;
    }

    @Override
    public void computeGPU(CUdeviceptr ptr, int size, CUstream stream) {
        CudaFunctions.LeakyReLU(ptr, size, alpha, CudaUtil.PREFERRED_BLOCK_SIZE, stream);
    }

    @Override
    public void computeMultiGPU(CUdeviceptr2D ptr, int width, int height, CUstream stream) {
        int device = CudaEngine.getThreadDeviceId();
        
        CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_ACTIVATION, "multi_ReLU", device);
        
        Pointer kernelParameters = Pointer.to(
            Pointer.to(ptr.ptr),
            Pointer.to(new long[]{width}),
            Pointer.to(new long[]{ptr.pitch})
        );
        
        int blockSizeX = Math.min(CudaEngine.getMaxThreadsPerBlock(device), width);
        int gridSizeX = (width - 1) / blockSizeX + 1;
        int gridSizeY = height;
        
        JCudaDriver.cuLaunchKernel(function,
            gridSizeX, gridSizeY, 1,   // Grid dimension
            blockSizeX, 1, 1,          // Block dimension
            0, stream,                 // Shared memory size and stream
            kernelParameters, null     // Kernel- and extra parameters
        );
    }

    @Override
    public void computeConditional(float[] input, boolean[] compute) {
        for(int i=0; i < input.length; i++)
            if(compute[i])
                input[i] = Math.max(0, input[i]);
    }

    @Override
    public void computeGPUConditional(CUdeviceptr ptr, CUdeviceptr compute, int size, CUstream stream, int count) {
        int device = CudaEngine.getThreadDeviceId();
        
        CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_ACTIVATION, "ReLU_Conditional", device);
        
        Pointer kernelParameters = Pointer.to(
            Pointer.to(ptr),
            Pointer.to(compute),
            Pointer.to(new long[]{size})
        );
        
        int blockSizeX = Math.min(CudaEngine.getMaxThreadsPerBlock(device), size);
        int gridSizeX = (size - 1) / blockSizeX + 1;
        
        JCudaDriver.cuLaunchKernel(function,
            gridSizeX, count, 1,    // Grid dimension
            blockSizeX, 1, 1,       // Block dimension
            0, stream,              // Shared memory size and stream
            kernelParameters, null  // Kernel- and extra parameters
        );
    }

    @Override
    public void computeMultiGPUConditional(CUdeviceptr2D ptr, CUdeviceptr compute, int width, int height, CUstream stream) {
        int device = CudaEngine.getThreadDeviceId();
        
        CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_ACTIVATION, "multi_ReLU_Conditional", device);
        
        Pointer kernelParameters = Pointer.to(
            Pointer.to(ptr.ptr),
            Pointer.to(compute),
            Pointer.to(new long[]{width}),
            Pointer.to(new long[]{ptr.pitch})
        );
        
        int blockSizeX = Math.min(CudaEngine.getMaxThreadsPerBlock(device), width);
        int gridSizeX = (width - 1) / blockSizeX + 1;
        int gridSizeY = height;
        
        JCudaDriver.cuLaunchKernel(function,
            gridSizeX, gridSizeY, 1,   // Grid dimension
            blockSizeX, 1, 1,          // Block dimension
            0, stream,                 // Shared memory size and stream
            kernelParameters, null     // Kernel- and extra parameters
        );
    }
    
    @Override
    public void derivative(float[] input, int from, int to) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public void compute(FloatBuffer input, int stride, int count) {
        throw new UnsupportedOperationException("Not supported yet.");
    }
}