use crate::grid_function::GridFunction;

impl GridFunction {
    /// # Forwards difference derivative
    ///
    /// ## Description
    /// Calculates the approximate derivative of a `GridFunction` using the
    /// forwards difference scheme, and returns the derivative as a new
    /// `GridFunction`.
    ///
    /// The derivative at the first and final grid points are
    /// approximated using a forwards difference and backwards difference scheme
    /// respectively.
    ///
    /// ## Example use case
    /// Suppose that we have a `GridFunction` `grid_func` and we want to
    /// calculate its derivative. The code below does this.
    /// ```
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
    /// let grid_func = GridFunction::new_grid_function(&grid, |x| x.powi(2));
    /// let grid_func_derivative = grid_func.forward_difference_derivative();
    /// ```
    ///
    pub fn forward_difference_derivative(self: &Self) -> Self {
        let grid = &self.grid;
        let function_values = &self.function_values;

        let grid_points = &grid.grid_points;
        let num_points = grid_points.len();

        let mut first_derivative_values = Vec::new();

        // Calculates the derivative at the starting grid point using the
        // forwards difference scheme.
        first_derivative_values.push(
            (function_values[1] - function_values[0])
                / (grid_points[1] - grid_points[0]),
        );

        // Calculates the derivative at each interior grid point using the forwards difference scheme.
        for i in 1..(num_points - 1) {
            first_derivative_values.push(
                (function_values[i + 1] - function_values[i])
                    / (grid_points[i + 1] - grid_points[i]),
            );
        }

        // Calculates the derivative at the final grid point using the backwards difference scheme.
        first_derivative_values.push(
            (function_values[num_points - 1] - function_values[num_points - 2])
                / (grid_points[num_points - 1] - grid_points[num_points - 2]),
        );

        GridFunction {
            grid: grid.clone(),
            function_values: first_derivative_values,
        }
    }

    /// # Central difference derivative
    ///
    /// ## Description
    ///
    /// Calculates the approximate derivative of a `GridFunction` using the
    /// central difference scheme, and returns the derivative as a new
    /// `GridFunction`.
    ///
    /// The derivative at the first and final grid points are
    /// approximated using a forwards difference and backwards difference scheme
    /// respectively.
    ///
    /// ## Example use case
    /// Suppose that we have a `GridFunction` `grid_func` and we want to
    /// calculate its derivative. The code below does this.
    /// ```
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
    /// let grid_func = GridFunction::new_grid_function(&grid, |x| x.powi(2));
    /// let grid_func_derivative = grid_func.central_difference_derivative();
    /// ```
    ///
    pub fn central_difference_derivative(self: &Self) -> Self {
        let grid = &self.grid;
        let function_values = &self.function_values;

        let grid_points = &grid.grid_points;
        let num_points = grid_points.len();

        let mut first_derivative_values = Vec::new();

        // Calculates the derivative at the starting grid point using the forwards difference scheme.
        first_derivative_values.push(
            (function_values[1] - function_values[0])
                / (grid_points[1] - grid_points[0]),
        );

        // Calculates the derivative at each interior grid point using the central difference scheme.
        for i in 1..(num_points - 1) {
            first_derivative_values.push(
                (function_values[i + 1] - function_values[i - 1])
                    / (grid_points[i + 1] - grid_points[i - 1]),
            );
        }

        // Calculates the derivative at the final grid point using the backwards difference scheme.
        first_derivative_values.push(
            (function_values[num_points - 1] - function_values[num_points - 2])
                / (grid_points[num_points - 1] - grid_points[num_points - 2]),
        );

        GridFunction {
            grid: grid.clone(),
            function_values: first_derivative_values,
        }
    }

    /// # Derivative
    ///
    /// ## Description
    /// This function performs the best differentiation scheme available for the
    /// `GridFunction` object.
    ///
    /// Currently, the best differentiation scheme available is the central
    /// difference scheme.
    ///
    /// ## Example use case
    /// Suppose that we have a `GridFunction` `grid_func` and we want to
    /// calculate its derivative. The code below does this.
    /// ```
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
    /// let grid_func = GridFunction::new_grid_function(&grid, |x| x.powi(2));
    /// let grid_func_derivative = grid_func.derivative();
    /// ```
    ///
    pub fn derivative(self: &Self) -> Self {
        // Performs the central difference derivative.
        self.central_difference_derivative()
    }
}
