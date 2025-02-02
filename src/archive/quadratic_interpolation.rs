/// # Quadratic interpolation
///
/// ## Description
/// Approximates a real-valued function of a real variable with a quadratic
/// polynomial, and returns the coefficients of the quadratic polynomial.
///
/// Three points and their associated function values are required to
/// approximate the function with a quadratic polynomial. The points are
/// specified as an array `points` and the function values are specified as
/// an array `function_values`. The coefficients of the quadratic polynomial are
/// returned as an array `[a, b, c]`, such that the quadratic has the form
/// `ax^2 + bx + c`.
///
/// ## Example use case
/// ```
/// let points = [0.0, 1.0, 2.0];
/// let function_values = [0.0, 1.0, 4.0];
/// let coefficients = quadratic_interpolation_coefficients(points,
///     function_values).unwrap();
/// ```
///
pub fn quadratic_interpolation_coefficients(
    points: [f64; 3],
    function_values: [f64; 3],
) -> Result<[f64; 3], &'static str> {
    let [x0, x1, x2] = points;
    let [f0, f1, f2] = function_values;

    if x0 == x1 || x0 == x2 || x1 == x2 {
        return Err("Points must be distinct");
    }

    let d0 = f0 / ((x0 - x1) * (x0 - x2));
    let d1 = f1 / ((x1 - x0) * (x1 - x2));
    let d2 = f2 / ((x2 - x0) * (x2 - x1));

    let a = d0 + d1 + d2;
    let b = -(d0 * (x1 + x2)) - (d1 * (x0 + x2)) - (d2 * (x0 + x1));
    let c = (d0 * x1 * x2) + (d1 * x0 * x2) + (d2 * x0 * x1);

    Ok([a, b, c])
}

/// # Quadratic integral
///
/// ## Description
/// Calculates the definite integral of a quadratic polynomial
/// `a*x^2 + b*x + c` from `lower_limit` to `upper_limit`.
///
/// The coefficients are specified as an array `coefficients` in the form
/// `[a, b, c]`.
///
/// ## Example use case
/// ```
/// let coefficients = (2.0, 3.0, 1.0);
/// integral = quadratic_integral(coefficients, 0.0, 1.0).unwrap();
/// ```
///
pub fn quadratic_integral(
    coefficients: [f64; 3],
    lower_limit: f64,
    upper_limit: f64,
) -> Result<f64, &'static str> {
    if lower_limit >= upper_limit {
        return Err("lower_limit must be less than upper_limit");
    }

    let [a, b, c] = coefficients;

    // Calculates the definite integral from 0 to upper_limit.
    let integral_upper = (a / 3.0) * upper_limit.powi(3)
        + (b / 2.0) * upper_limit.powi(2)
        + c * upper_limit;

    // Calculates the definite integral from 0 to lower_limit.
    let integral_lower = (a / 3.0) * lower_limit.powi(3)
        + (b / 2.0) * lower_limit.powi(2)
        + c * lower_limit;

    // Evaluates the definite integral from lower_limit to upper_limit.
    Ok(integral_upper - integral_lower)
}

/// # Quadratic derivative
///
/// ## Description
/// Calculates the derivative of a quadratic polynomial `a*x^2 + b*x + c` at a
/// given coordinate.
///
/// The coefficients are specified as an array `coefficients` in the form
/// `[a, b, c]`.
///
/// ## Example use case
/// ```
/// use crate::quadratic_interpolation;
///
/// let coefficients = [2.0, 3.0, 1.0];
/// let coordinate = 1.0;
/// let derivative_value = quadratic_interpolation::quadratic_derivative(coefficients, coordinate);
/// ```
///
pub fn quadratic_derivative(coefficients: [f64; 3], coordinate: f64) -> f64 {
    let [a, b, _] = coefficients;
    (2.0 * a * coordinate) + b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quadratic_interpolation_coefficients() {
        // Test with a constant function
        let points = [0.0, 1.0, 2.0];
        let function_values = [1.0, 1.0, 1.0];
        let coefficients =
            quadratic_interpolation_coefficients(points, function_values)
                .unwrap();
        assert_eq!(coefficients, [0.0, 0.0, 1.0]);

        // Test with the function x.
        let points = [0.0, 1.0, 2.0];
        let function_values = [0.0, 1.0, 2.0];
        let coefficients =
            quadratic_interpolation_coefficients(points, function_values)
                .unwrap();
        assert_eq!(coefficients, [0.0, 1.0, 0.0]);

        // Test with the function x^2.
        let points = [0.0, 1.0, 2.0];
        let function_values = [0.0, 1.0, 4.0];
        let coefficients =
            quadratic_interpolation_coefficients(points, function_values)
                .unwrap();
        assert_eq!(coefficients, [1.0, 0.0, 0.0]);

        // Test with a set of points that don't start at 0.
        let points = [1.0, 2.0, 3.0];
        let function_values = [1.0, 4.0, 9.0];
        let coefficients =
            quadratic_interpolation_coefficients(points, function_values)
                .unwrap();
        assert_eq!(coefficients, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_quadratic_integral() {
        // Test with a constant function
        let coefficients = [0.0, 0.0, 1.0];
        let integral = quadratic_integral(coefficients, 0.0, 1.0).unwrap();
        assert_eq!(integral, 1.0);

        // Test with the function x
        let coefficients = [0.0, 1.0, 0.0];
        let integral = quadratic_integral(coefficients, 0.0, 1.0).unwrap();
        assert!((integral - 0.5).abs() < 1e-6);

        // Test with the function x^2
        let coefficients = [1.0, 0.0, 0.0];
        let integral = quadratic_integral(coefficients, 0.0, 1.0).unwrap();
        assert!((integral - 0.3333333333333333).abs() < 1e-6);

        // Test with a set of points that don't start at 0.
        let coefficients = [1.0, 2.0, 3.0];
        let integral = quadratic_integral(coefficients, 1.0, 2.0).unwrap();
        assert!((integral - 8.33333333333334).abs() < 1e-6);
    }

    #[test]
    fn test_quadratic_derivative() {
        // Test with a constant function
        let coefficients = [0.0, 0.0, 1.0];
        let coordinate = 0.0;
        let derivative_value = quadratic_derivative(coefficients, coordinate);
        assert!((derivative_value - 0.0).abs() < 1e-6);

        // Test with the function x
        let coefficients = [0.0, 1.0, 0.0];
        let coordinate = 0.0;
        let derivative_value = quadratic_derivative(coefficients, coordinate);
        assert!((derivative_value - 1.0).abs() < 1e-6);

        // Test with the function x^2
        let coefficients = [1.0, 0.0, 0.0];
        let coordinate = 0.0;
        let derivative_value = quadratic_derivative(coefficients, coordinate);
        assert!((derivative_value - 0.0).abs() < 1e-6);

        // Test with a non-zero coordinate
        let coefficients = [1.0, 0.0, 0.0];
        let coordinate = 1.0;
        let derivative_value = quadratic_derivative(coefficients, coordinate);
        assert!((derivative_value - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_quadratic_interpolation_coefficients_error() {
        // Test with duplicate points
        let points = [0.0, 1.0, 1.0];
        let function_values = [0.0, 1.0, 2.0];
        let result =
            quadratic_interpolation_coefficients(points, function_values);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Points must be distinct");

        // Test with another set of duplicate points
        let points = [2.0, 2.0, 3.0];
        let function_values = [4.0, 4.0, 9.0];
        let result =
            quadratic_interpolation_coefficients(points, function_values);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Points must be distinct");
    }

    #[test]
    fn test_quadratic_integral_error() {
        // Test with lower_limit greater than upper_limit
        let coefficients = [1.0, 2.0, 3.0];
        let result = quadratic_integral(coefficients, 2.0, 1.0);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "lower_limit must be less than upper_limit"
        );

        // Test with lower_limit equal to upper_limit
        let result = quadratic_integral(coefficients, 1.0, 1.0);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "lower_limit must be less than upper_limit"
        );
    }
}
