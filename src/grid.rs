/// # Grid
///
/// ## Description
/// `Grid` represents a grid of points in 1D. The coordinate of each point
/// corresponds to an element in the vector `grid_points`.
///
/// ## Example use case
/// ```
/// let grid_points = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
/// let grid = Grid { grid_points };
/// ```
///
/// ## Todo
/// Add functionality to create non-uniform grids.
/// Change implementation so uniform grids are stored as metadata, rather than
/// storing the grid points directly.
///
#[derive(Debug, Clone, PartialEq)]
pub struct Grid {
    pub grid_points: Vec<f64>,
}

impl Grid {
    /// # New uniform grid
    ///
    /// ## Description
    /// `new_uniform_grid` creates a uniform `Grid` of `num_points` points
    /// between `start_point` and `end_point` inclusive.
    ///
    /// If `start_point` is greater than or equal to `end_point` or `num_points`
    /// is equal to 0, the function returns an empty vector.
    ///
    /// If `num_points` is equal to 1, the function returns a vector containing
    /// only `start_point`.
    ///
    /// ## Example use case
    /// Suppose that we want to create a uniform grid with 11 points between
    /// 0.0 and 1.0 inclusive. The code below does this.
    /// ```
    /// let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
    /// ```
    ///
    pub fn new_uniform_grid(
        start_point: f64,
        end_point: f64,
        num_points: usize,
    ) -> Self {
        // Error handling for edge case when start_point is greater than or
        // equal to end_point or num_points is equal to 0.
        // For this case, the function returns an empty vector.
        if start_point >= end_point || num_points == 0 {
            let grid_points: Vec<f64> = vec![];
            return Grid { grid_points };
        }

        // Error handling for edge case when num_points is equal to 1.
        // For this case, the function returns a vector containing only
        // start_point.
        if num_points == 1 {
            let grid_points: Vec<f64> = vec![start_point];
            return Grid { grid_points };
        }

        // step_size is the distance between adjacent grid points
        let step_size = (end_point - start_point) / (num_points as f64 - 1.0);

        // Creates a vector containing the grid points
        let grid_points: Vec<f64> = (0..num_points)
            .map(|i| start_point + (i as f64) * step_size)
            .collect();

        Grid { grid_points }
    }
}

/// ## Todo
/// Ensure that modifying a clone does not affect the original.
/// Add tests which test floating point precision. For example, a grid with  
/// start_point and end_point close together, or a grid with a large number of
/// points.
/// Test construction of a non-uniform grid, once this functionality is added.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_uniform_grid() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 11);

        for (a, b) in grid.grid_points.iter().zip(
            vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].iter(),
        ) {
            assert!((a - b).abs() < 1e-10, "new_uniform_grid failed to create a grid of 11 points with sufficient precision.");
        }
    }

    #[test]
    fn test_grid_debug() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
        let debug_str = format!("{:?}", grid);
        assert!(
            debug_str.contains("Grid { grid_points: ["),
            "Debug trait implementation is incorrect."
        );
    }

    #[test]
    fn test_grid_clone() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 11);
        let cloned_grid = grid.clone();
        assert_eq!(
            grid, cloned_grid,
            "Clone trait implementation is incorrect."
        );
    }

    #[test]
    fn test_grid_partial_eq() {
        let grid = Grid::new_uniform_grid(0.0, 1.0, 11);

        // Test equality with itself.
        assert_eq!(grid, grid, "PartialEq trait implementation is incorrect. Equality of grid with itself failed.");

        // Test equality with a clone.
        let grid_clone = grid.clone();
        assert_eq!(grid, grid_clone, "PartialEq trait implementation is incorrect. Equality of grid with a clone failed.");

        // Test inequality with a different grid.
        let grid_other = Grid::new_uniform_grid(0.0, 1.0, 10);
        assert_ne!(
            grid, grid_other,
            "PartialEq trait implementation is incorrect. Inequality of grid with a different grid failed."
        );

        // Test inequality with a different grid.
        let grid = Grid::new_uniform_grid(0.0, 1.0, 2);
        let grid_other = Grid::new_uniform_grid(0.0, 1.0, 3);
        assert_ne!(
            grid, grid_other,
            "PartialEq trait implementation is incorrect. Inequality of grid with a different grid failed."
        );
    }

    #[test]
    fn test_new_uniform_grid_edge_cases() {
        // Test with a single grid point.
        let grid = Grid::new_uniform_grid(0.0, 1.0, 1);
        assert_eq!(grid.grid_points, vec![0.0], "new_uniform_grid failed to handle edge case with a single grid point.");

        // Test with no grid points.
        let grid = Grid::new_uniform_grid(0.0, 1.0, 0);
        assert_eq!(
            grid.grid_points,
            vec![],
            "new_uniform_grid failed to handle edge case with no grid points."
        );

        // Test with start_point equal to end_point.
        let grid = Grid::new_uniform_grid(1.0, 1.0, 11);
        assert_eq!(
            grid.grid_points,
            vec![],
            "new_uniform_grid failed to handle edge case with start_point equal to end_point."
        );

        // Test with start_point greater than end_point.
        let grid = Grid::new_uniform_grid(1.0, 0.0, 11);
        assert_eq!(
            grid.grid_points,
            vec![],
            "new_uniform_grid failed to handle edge case with start_point greater than end_point."
        );
    }
}
