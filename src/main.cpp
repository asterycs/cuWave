#include <iostream>
#include <exception>

#include "cxxopts.hpp"

#include "App.hpp"

int main(int argc, char * argv[]) {

  bool batch_render = false;

  cxxopts::Options options(argv[0], "");

  options.add_options()
    ("b,batch",     "Batch render",         cxxopts::value<bool>(batch_render))
    ("p,passes",    "Number of passes",      cxxopts::value<int>())
    ("s,scene",     "Scene file",           cxxopts::value<std::string>(),  "FILE")
    ("o,output",    "Output file",          cxxopts::value<std::string>(),  "FILE");



    auto optres = options.parse(argc, argv);


    if (batch_render)
    {
      if (!optres.count("scene"))
      {
        std::cerr << "No scene file specified" << std::endl;
        return 1;
      }

      if (!optres.count("output"))
      {
        std::cerr << "No output file specified" << std::endl;
        return 1;
      }

      if (!optres.count("passes"))
      {
        std::cerr << "Number of passes not specified" << std::endl;
        return 1;
      }

      std::string scenefile = optres["scene"].as<std::string>();
      std::string output = optres["output"].as<std::string>();
      int passes = optres["passes"].as<int>();

      try
      {
        App& app = App::getInstance();

	    app.pathTraceToFile(scenefile, output, passes);

      }
      catch (std::exception& e)
      {
        std::cout << e.what() << std::endl;
      }

  }else{

    try
    {
      App& app = App::getInstance();

      if (fileExists(LAST_SCENEFILE_NAME))
        app.loadSceneFile(LAST_SCENEFILE_NAME);

      app.showWindow();
      app.MainLoop();
    }
    catch (std::exception& e)
    {
      std::cout << e.what() << std::endl;
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}
