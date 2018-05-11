#include "Utils.hpp"
#include "Model.hpp"

#include <fstream>

#include <GL/glew.h>
#include <GL/gl.h>

#define ILUT_USE_OPENGL
#include <IL/ilu.h>

#ifdef ENABLE_CUDA
  #include <cuda_runtime.h>
#endif

#include "vector_math.hpp"

CUDA_HOST_DEVICE float3 glm42float3(const glm::fvec4 g)
{
	return make_float3(g.x, g.y, g.z);
}

CUDA_HOST_DEVICE float3 glm32float3(const glm::fvec3 g)
{
	return make_float3(g.x, g.y, g.z);
}

void CheckOpenGLError(const char* call, const char* fname, int line)
{
  GLenum error = glGetError();

  if (error != GL_NO_ERROR)
  {
    std::string errorStr;
    switch (error)
    {
      case GL_INVALID_ENUM:                   errorStr = "GL_INVALID_ENUM"; break;
      case GL_INVALID_VALUE:                  errorStr = "GL_INVALID_VALUE"; break;
      case GL_INVALID_OPERATION:              errorStr = "GL_INVALID_OPERATION"; break;
      case GL_STACK_OVERFLOW:                 errorStr = "GL_STACK_OVERFLOW"; break;
      case GL_STACK_UNDERFLOW:                errorStr = "GL_STACK_UNDERFLOW"; break;
      case GL_OUT_OF_MEMORY:                  errorStr = "GL_OUT_OF_MEMORY"; break;
      case GL_INVALID_FRAMEBUFFER_OPERATION:  errorStr = "GL_INVALID_FRAMEBUFFER_OPERATION"; break;
      default:                                errorStr = "Unknown error"; break;
    }

    std::cerr << "At: " << fname << ":" << line << std::endl \
     << " OpenGL call: " << call << std::endl \
      << " Error: " << errorStr << std::endl;
  }
}

void CheckILError(const char* call, const char* fname, int line)
{
  ILenum error = ilGetError();

  if (error != IL_NO_ERROR) {
    do {
      std::string errStr = iluErrorString(error);

      std::cerr << "At: " << fname << ":" << line << std::endl \
       << " IL call: " << call << std::endl \
        << " Error: " << errStr << std::endl;

    } while ((error = ilGetError ()));
  }
}

void CheckCudaError(const char* call, const char* fname, int line)
{
    cudaError_t result_ = cudaGetLastError();
    if (result_ != cudaSuccess) {
        std::cerr << "At: " << fname << ":" << line << std::endl \
           << " Cuda call: " << call << " Error: " << cudaGetErrorString(result_) << std::endl;
        throw std::runtime_error("Error in CUDA call");
    }
}

void CheckCurandError(const curandStatus_t status, const char* fname, int line)
{
    if (status != CURAND_STATUS_SUCCESS) {
        std::cerr << "Curand error at: " << fname << ":" << line << " ";

        switch (status)
        {
        	case CURAND_STATUS_VERSION_MISMATCH:
				std::cerr << "Header file and linked library version do not match" << std::endl;
				break;

        	case CURAND_STATUS_NOT_INITIALIZED:
				std::cerr << "Generator not initialized" << std::endl;
				break;

        	case CURAND_STATUS_ALLOCATION_FAILED:
				std::cerr << "Memory allocation failed" << std::endl;
				break;

        	case CURAND_STATUS_TYPE_ERROR:
				std::cerr << "Generator is wrong type" << std::endl;
				break;

        	case CURAND_STATUS_OUT_OF_RANGE:
				std::cerr << "Argument out of range" << std::endl;
				break;

        	case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
				std::cerr << "Length requested is not a multple of dimension" << std::endl;
				break;

        	case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
        		std::cerr << "GPU does not have double precision required by MRG32k3a" << std::endl;
        		break;

        	case CURAND_STATUS_LAUNCH_FAILURE:
        		std::cerr << "Kernel launch failure" << std::endl;
        		break;

        	case CURAND_STATUS_PREEXISTING_FAILURE:
        		std::cerr << "Preexisting failure on library entry" << std::endl;
        		break;

        	case CURAND_STATUS_INITIALIZATION_FAILED:
        		std::cerr << "Initialization of CUDA failed" << std::endl;
        		break;

        	case CURAND_STATUS_ARCH_MISMATCH:
        		std::cerr << "Architecture mismatch, GPU does not support requested feature" << std::endl;
        		break;

        	case CURAND_STATUS_INTERNAL_ERROR:
        		std::cerr << "Internal error" << std::endl;
        		break;

        	default:
        		std::cerr << "Unknown error" << std::endl;
        }
        throw std::runtime_error("Error in cuRAND call");
    }
}

std::string readFile(const std::string& filePath) {
    std::string content;
    std::ifstream fileStream(filePath, std::ios::in);

    if(!fileStream.is_open()) {
        std::cerr << "Could not read file " << filePath << ". File does not exist." << std::endl;
        return "";
    }

    std::string line = "";
    while(!fileStream.eof()) {
        std::getline(fileStream, line);
        content.append(line + "\n");
    }

    fileStream.close();
    return content;
}

bool fileExists(const std::string& filename)
{
    std::ifstream infile(filename);
    return infile.good();
}

CUDA_HOST_DEVICE float AABB::area() const
{
  float3 d = max - min;

  return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
}

CUDA_HOST_DEVICE unsigned int AABB::maxAxis() const
{
  const float3 d = abs(max - min);

  if (d.x >= d.y && d.x >= d.z)
    return 0;
  else if (d.y >= d.x && d.y >= d.z)
    return 1;
  else
    return 2;
}

CUDA_HOST_DEVICE void AABB::add(const Triangle& t)
{
  for (auto& v : t.vertices)
    add(v.p);
}

CUDA_HOST_DEVICE void AABB::add(const float3 v)
{
  min = fmin(min, v);
  max = fmax(max, v);
}
