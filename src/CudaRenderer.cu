#include "CudaRenderer.hpp"

#include <GL/glew.h>
#include <GL/gl.h>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math_constants.h>
#include <cooperative_groups.h>

#include "Utils.hpp"
#include "Triangle.hpp"
#include "Kernels.hpp"

void CudaRenderer::reset()
{
	queues.reset();
	callcntr = 0;

	dim3 block(BLOCKWIDTH, BLOCKWIDTH);
	dim3 grid((lastSize.x + block.x - 1) / block.x, (lastSize.y + block.y - 1) / block.y);

	resetAllPaths<<<grid, block>>>(paths, lastCamera, lastSize);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void CudaRenderer::resize(const glm::ivec2 size)
{
	queues.resize(size);
	paths.resize(size);

	lastSize = size;

	dim3 block(BLOCKWIDTH, BLOCKWIDTH);
	dim3 grid((size.x + block.x - 1) / block.x,
			(size.y + block.y - 1) / block.y);

	uint32_t* hostScrambleConstants;

	// Seems there are only 20 000 scramble constants available. We'll reuse them.
	CURAND_CHECK(curandGetScrambleConstants32(&hostScrambleConstants));

	const int jump = 20000;
	for (int startIdx = 0; startIdx < size.x*size.y; startIdx += jump)
	{
		const int toCopy = std::min(jump, size.x*size.y-startIdx);
		CUDA_CHECK(cudaMemcpy(paths.scrambleConstants + startIdx, hostScrambleConstants, toCopy*sizeof(uint32_t), cudaMemcpyHostToDevice));
	}

	reset();
}

CudaRenderer::CudaRenderer() :
		lastCamera(), lastSize(), callcntr(0)
{
	uint32_t cudaDeviceCount = 0;
	int cudaDevices[8];
	uint32_t cudaDevicesCount = 8;

	cudaGLGetDevices(&cudaDeviceCount, cudaDevices, cudaDevicesCount,
			cudaGLDeviceListCurrentFrame);

	if (cudaDeviceCount < 1)
		throw std::runtime_error("No CUDA devices available");


	CURAND_CHECK(curandCreateGenerator(&randGen, CURAND_RNG_QUASI_SOBOL32));
	CURAND_CHECK(curandSetQuasiRandomGeneratorDimensions(randGen, RANDOM_DIMENSIONS));

	CUDA_CHECK(cudaSetDevice(cudaDevices[0]));

	resize(glm::ivec2(WWIDTH, WHEIGHT));
}

CudaRenderer::~CudaRenderer()
{
	CURAND_CHECK(curandDestroyGenerator(randGen));
	queues.release();
	paths.release();
}

void CudaRenderer::pathTraceToCanvas(GLTexture& canvas, const Camera& camera,
		CudaModel& model)
{
	if (model.getNTriangles() == 0)
		return;

	const glm::ivec2 canvasSize = canvas.getSize();
	const bool diffCamera = std::memcmp(&camera, &lastCamera, sizeof(Camera));
	const bool diffSize = (canvasSize != lastSize);
	const auto surfaceObj = canvas.getCudaMappedSurfaceObject();

	const dim3 block(BLOCKWIDTH, BLOCKWIDTH);
	const dim3 grid((canvasSize.x + block.x - 1) / block.x,
			(canvasSize.y + block.y - 1) / block.y);

	if (diffCamera != 0 || diffSize != 0)
	{
		lastCamera = camera;
		reset();
	}

    CURAND_CHECK(curandGenerateUniform(randGen, paths.floats, RANDOM_DIMENSIONS));

	castRays<<<grid, block>>>(paths, canvasSize, model.getDeviceTriangles(),
			model.getDeviceBVH(), model.getDeviceMaterials(),
			model.getDeviceTriangleMaterialIds());

	CUDA_CHECK(cudaDeviceSynchronize());

	logicKernel<<<grid, block>>>(canvasSize, queues, paths,
			model.getDeviceMaterials(), model.getDeviceTriangleMaterialIds(), model.getDeviceTriangles());

	CUDA_CHECK(cudaDeviceSynchronize());

	diffuseKernel<<<grid, block>>>(canvasSize, queues, paths,
			model.getDeviceTriangles(), model.getDeviceLightIds(),
			model.getNLights(), model.getDeviceTriangleMaterialIds(),
			model.getDeviceMaterials(), model.getDeviceBVH());

	CUDA_CHECK(cudaDeviceSynchronize());

	specularExtensionKernel<<<grid, block>>>(
		canvasSize,
		queues,
		paths,
		model.getDeviceTriangles()
	);

	 CUDA_CHECK(cudaDeviceSynchronize());
	 CUDA_CHECK(cudaMemset(queues.specularQueueSize, 0, sizeof(uint32_t)));

	transparentExtensionKernel<<<grid, block>>>(
		canvasSize,
		queues,
		paths,
		model.getDeviceTriangles(),
		model.getDeviceTriangleMaterialIds(),
		model.getDeviceMaterials()
	 );

	 CUDA_CHECK(cudaDeviceSynchronize());
	 CUDA_CHECK(cudaMemset(queues.transparentQueueSize, 0, sizeof(uint32_t)));

	writeToCanvas<<<grid, block>>>(canvasSize, surfaceObj, paths);

	diffuseExtensionKernel<<<grid, block>>>(canvasSize, queues, paths,
			model.getDeviceTriangles(), model.getDeviceTriangleMaterialIds(),
			model.getDeviceMaterials(), model.getNLights());

	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaMemset(queues.diffuseQueueSize, 0, sizeof(uint32_t)));

	newPathsKernel<<<grid, block>>>(canvasSize, queues, paths, camera);

	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaMemset(queues.newPathQueueSize, 0, sizeof(uint32_t)));

	canvas.cudaUnmap();
}

