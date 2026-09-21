#pragma once

#include <memory>
#include <functional>
#include "datatypes.hpp"

class Variable;
class FieldQuantity;
class Grid;
class System;

class DynamicEquation {
 public:
  DynamicEquation(const Variable* x,
                  std::shared_ptr<FieldQuantity> rhs,
                  std::shared_ptr<FieldQuantity> noiseTerm = nullptr);

  const Variable* x;
  std::shared_ptr<FieldQuantity> rhs;
  std::shared_ptr<FieldQuantity> noiseTerm;

  int ncomp() const;
  Grid grid() const;
  std::shared_ptr<const System> system() const;
  std::function<real()> maxOmega = nullptr;
};
