#include <iostream>

#include "application.h"
#include "cli_options.h"

int
main(int argc, char** argv)
{
    auto settings = chibillm::parse_cli_options(argc, argv);
    if (!settings) {
        std::cerr << settings.error() << '\n';
        chibillm::print_usage(std::cerr);
        return 1;
    }
    if (settings->help) {
        chibillm::print_usage(std::cout);
        return 0;
    }
    return chibillm::run_application(*settings, CHIBILLM_SHADER_PATH);
}
