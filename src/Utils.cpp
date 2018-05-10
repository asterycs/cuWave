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
        exit(1);
    }
}

void CheckCurandError(const curandStatus_t status, const char* fname, int line)
{
    if (status != CURAND_STATUS_SUCCESS) {
        std::cerr << "Curand error at: " << fname << ":" << line << std::endl;
        exit(1);
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
