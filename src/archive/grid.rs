use crate::errors::GridError;

/// # Grid 1D
///
/// ## Description
/// `Grid1D` represents a grid of points in 1D, which can be either uniform or
/// non-uniform.
///
#[derive(Debug, Clone, PartialEq)]
pub enum Grid1D {
    Uniform(f64, f64, usize),
    NonUniform(Vec<f64>),
}

// Constructor functions.
impl Grid1D {
    /// # New uniform grid
    ///
    /// ## Description
    /// Creates a new instance of `Grid1D`, representing a uniform grid.
    ///
    pub fn new_uniform_grid(
        start_point: f64,
        end_point: f64,
        num_points: usize,
    ) -> Result<Self, GridError> {
        // Error handling for invalid range.
        if start_point > end_point {
            return Err(GridError::InvalidRange);
        }

        // Error handling for zero points.
        if num_points <= 1 {
            return Err(GridError::InvalidNumPoints);
        }

        Ok(Grid1D::Uniform(start_point, end_point, num_points))
    }
}

// Methods.
impl Grid1D {
    /// # Num points
    ///
    /// ## Description
    /// Returns the number of grid points given a `Grid1D` instance.
    ///
    pub fn num_points(&self) -> usize {
        match self {
            Grid1D::Uniform(_, _, num_points) => *num_points,
            Grid1D::NonUniform(points) => points.len(),
        }
    }

    /// # Grid points
    ///
    /// ## Description
    /// Returns the grid points as a vector of f64 values given an instance of
    /// `Grid`.
    ///
    pub fn grid_points(&self) -> Vec<f64> {
        match self {
            Grid::Uniform(start_point, end_point, num_points) => {
                let step =
                    (end_point - start_point) / (*num_points as f64 - 1.0);
                (0..*num_points)
                    .map(|i| start_point + i as f64 * step)
                    .collect()
            }
            Grid::NonUniform(points) => points.clone(),
        }
    }
}

// Tests.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils;

    mod grid_traits_tests {

        use super::*;

        #[test]
        fn test_grid_debug() {
            let grid = Grid::new_uniform_grid(0.0, 1.0, 11).unwrap();
            assert_eq!(
                format!("{:?}", grid),
                "Uniform(0.0, 1.0, 11)",
                "Debug trait implementation is incorrect."
            );
        }

        #[test]
        fn test_grid_clone() {
            let grid = Grid::new_uniform_grid(0.0, 1.0, 11).unwrap();
            let grid_clone = grid.clone();
            assert_eq!(
                grid, grid_clone,
                "Clone trait implementation is incorrect. Cloning failed."
            );
        }

        #[test]
        fn test_grid_partial_eq() {
            let grid = Grid::new_uniform_grid(0.0, 1.0, 11).unwrap();

            // Test equality with itself.
            assert_eq!(grid,grid,
                "PartialEq trait implementation is incorrect. Equality of grid with itself failed."
            );

            // Test equality with a clone.
            let grid_clone = grid.clone();
            assert_eq!(grid, grid_clone, "PartialEq trait implementation is incorrect. Equality of grid with a clone failed.");

            // Test inequality with a different grid.
            let grid_other = Grid::new_uniform_grid(0.0, 1.0, 10).unwrap();
            assert_ne!(
                grid, grid_other, "PartialEq trait implementation is incorrect. Inequality of grid with a different grid failed."
            );
        }
    }

    mod constructor_functions_tests {
        use super::*;

        #[test]
        fn test_new_uniform_grid() {
            let grid = Grid::new_uniform_grid(0.0, 1.0, 11).unwrap();
            assert_eq!(
                grid,
                Grid::Uniform(0.0, 1.0, 11),
                "new_uniform_grid failed with valid arguments."
            );
        }

        #[test]
        fn test_new_uniform_grid_error_handling() {
            // Test with invalid range.
            let grid = Grid::new_uniform_grid(1.0, 0.0, 11);
            assert!(
                matches!(grid, Err(GridError::InvalidRange)),
                "new_uniform_grid failed to handle invalid range."
            );

            // Test with zero points.
            let grid = Grid::new_uniform_grid(0.0, 1.0, 0);
            assert!(
                matches!(grid, Err(GridError::InvalidNumPoints)),
                "new_uniform_grid failed to handle zero points."
            );
        }
    }

    mod methods_tests {
        use super::*;

        #[test]
        fn test_num_points() {
            let grid = Grid::new_uniform_grid(0.0, 1.0, 11).unwrap();
            assert_eq!(
                grid.num_points(),
                11,
                "num_points failed for a uniform grid."
            );

            let grid = Grid::NonUniform(vec![0.0, 0.1, 0.2, 0.3, 0.4]);
            assert_eq!(
                grid.num_points(),
                5,
                "num_points failed for a non-uniform grid."
            );
        }

        #[test]
        fn test_grid_points() {
            let grid = Grid::new_uniform_grid(0.0, 1.0, 11).unwrap();
            assert!(
                utils::vector_equality(
                    &grid.grid_points(),
                    &vec![
                        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
                    ],
                    1e-6
                ),
                "grid_points failed for a uniform grid."
            );

            let grid = Grid::NonUniform(vec![0.0, 0.1, 0.2, 0.3, 0.4]);
            assert_eq!(
                grid.grid_points(),
                vec![0.0, 0.1, 0.2, 0.3, 0.4],
                "grid_points failed for a non-uniform grid."
            );
        }
    }
}
