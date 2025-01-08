use crate::grid_function::GridFunction;
use crate::quadratic_interpolation;

// Fixed step size numerical integration algorithms.
impl GridFunction {
    /// # Riemann sum numerical integration algorithm.
    ///
    /// ## Description
    /// `integrate_riemann_sum` approximates the definite integral of a
    /// real-valued function of a real variable. This function is represented
    /// by a `GridFunction`. The lower and upper limits of the integral are
    /// the first and last grid points of the `GridFunction`.
    ///
    /// Over each grid cell, the function is approximated as being flat. The
    /// integral over each grid cell is computed, and all of these integrals are
    /// summed to approximate the total integral.
    ///
    /// ## Example use case
    /// Suppose we want to calculate the integral of the function `f(x) = x^2`
    /// from `x = 0` to `x = 1`. We can represent this function as a
    /// `GridFunction` and calculate the integral with the code below.
    /// ```
    /// let grid = Grid::new(0.0, 1.0, 11);
    /// let grid_func = GridFunction::new_grid_function(&grid, |x| x.powi(2));
    /// let integral = grid_func.integrate_riemann_sum();
    /// ```
    ///
    pub fn integrate_riemann_sum(self: &Self) -> f64 {
        let num_points = &self.grid.grid_points.len();
        let grid_points = &self.grid.grid_points;
        let function_values = &self.function_values;

        let mut integral = 0.0;

        for n in 0..(num_points - 1) {
            // Adding the contribution to the integral from each grid cell.
            // The function is approximated as being flat over each grid cell.
            integral +=
                function_values[n] * (grid_points[n + 1] - grid_points[n]);
        }

        integral
    }

    /// # Simpson's composite rule numerical integration algorithm.
    ///
    /// ## Description
    /// `integrate_composite_simpsons_rule` approximates the definite integral
    /// of a real-valued function of a real variable. This function is
    /// represented by a `GridFunction`. The lower and upper limits of the
    /// integral are the first and last grid points of the `GridFunction`.
    ///
    /// Over each pair of grid cells, the function is approximated as being
    /// quadratic. The integral over each pair of grid cells is computed, and
    /// all of these integrals are summed to approximate the total integral.
    ///
    /// The number of grid points must be odd.
    ///
    /// ## Example use case
    /// Suppose we want to calculate the integral of the function `f(x) = x^2`
    /// from `x = 0` to `x = 1`. We can represent this function as a
    /// `GridFunction` and calculate the integral with the code below.
    /// ```
    /// let grid = Grid::new(0.0, 1.0, 11);
    /// let grid_func = GridFunction::new_grid_function(&grid, |x| x.powi(2));
    /// let integral = grid_func.integrate_composite_simpsons_rule();
    /// ```
    ///
    /// ## Todo
    /// Modify the algorithm so that it can handle the case where the number of
    /// grid points is even.
    ///
    pub fn integrate_composite_simpsons_rule(self: &Self) -> f64 {
        let num_points = &self.grid.grid_points.len();
        let grid_points = &self.grid.grid_points;
        let function_values = &self.function_values;

        // Error handling - the number of grid points must be odd.
        if num_points % 2 == 0 {
            panic!("Integral failed to evaluate. The number of grid points must be odd.");
        }

        let mut integral = 0.0;

        for n in (0..(num_points - 2)).step_by(2) {
            // Adding the contribution to the integral from each pair of grid
            // cells.
            // The function is approximated as being quadratic over each pair
            // of grid cells.
            let points =
                (grid_points[n], grid_points[n + 1], grid_points[n + 2]);
            let function_values = (
                function_values[n],
                function_values[n + 1],
                function_values[n + 2],
            );

            let quadratic_coefficients =
                quadratic_interpolation::quadratic_interpolation_coefficients(
                    points,
                    function_values,
                );

            integral += quadratic_interpolation::quadratic_integral(
                quadratic_coefficients,
                points.0,
                points.2,
            );
        }

        integral
    }
}
