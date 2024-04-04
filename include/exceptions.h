#ifndef PARTICLE_SIM_EXCEPTIONS_H
#define PARTICLE_SIM_EXCEPTIONS_H


#include <exception>

class InvalidNumberOfArguments : public std::exception {
    const char* what() const noexcept override {
            return "You must provide at least one argument.";
    }
};

class InvalidArgumentType : public std::exception {
    const char* what() const noexcept override {
            return "You must provide int arguments.";
    }
};

#endif //PARTICLE_SIM_EXCEPTIONS_H
