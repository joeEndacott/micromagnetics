/// # Quadratic interpolation
///
/// ## Description
/// `quadratic_interpolation_coefficients` approximates a real-valued function
/// of a real variable with a quadratic polynomial, and returns the
/// coefficients of the quadratic polynomial.
///
/// ## Example use case
/// Suppose we have three points (0, 1, 2) and the function values at those
/// points are (0, 1, 4). We can use `quadratic_interpolation_coefficients` to
/// approximate the function with a quadratic polynomial and get the
/// coefficients of the polynomial. We can then use these coefficients to
/// evaluate the polynomial at other points.
/// ```
/// let points = (0.0, 1.0, 2.0);
/// let function_values = (0.0, 1.0, 4.0);
/// let (a, b, c) = quadratic_interpolation_coefficients(points,
///     function_values);
/// ```
///
/// ## Todo
/// This function is not currently working. I'm still troubleshooting.
///
pub fn quadratic_interpolation_coefficients(
    points: (f64, f64, f64),
    function_values: (f64, f64, f64),
) -> (f64, f64, f64) {
    let (x0, x1, x2) = points;
    let (f0, f1, f2) = function_values;

    let d0 = f0 / ((x0 - x1) * (x0 - x2));
    let d1 = f1 / ((x1 - x0) * (x1 - x2));
    let d2 = f2 / ((x2 - x0) * (x2 - x1));

    let a = d0 + d1 + d2;
    let b = -(d0 * (x1 + x2)) - (d1 * (x0 + x2)) - (d2 * (x0 + x1));
    let c = (d0 * x1 * x2) + (d1 * x0 * x2) + (d2 * x0 * x1);

    (a, b, c)
}

/// # Quadratic integral
///
/// ## Description
/// `quadratic_integral` calculates the definite integral of a quadratic
/// polynomial `a*x^2 + b*x + c` from `lower_limit` to `upper_limit`.
///
/// `quadratic_integral` takes a tuple `(a, b, c)` of the coefficients of the
/// quadratic polynomial, and the `lower_limit` and `upper_limit` of the
/// definite integral.
///
/// ## Example use case
/// Suppose we have a quadratic polynomial `f(x) = 2*x^2 + 3*x + 1` and we want
/// to calculate the integral of this polynomial from `x = 0` to `x = 1`. We can
/// use `quadratic_integral` to calculate the integral with the code below.
/// ```
/// let coefficients = (2.0, 3.0, 1.0);
/// integral = quadratic_integral(coefficients, 0.0, 1.0);
/// ```
///
/// ## Todo
/// This function is not currently working. I'm still troubleshooting.
///
pub fn quadratic_integral(
    coefficients: (f64, f64, f64),
    lower_limit: f64,
    upper_limit: f64,
) -> f64 {
    let (a, b, c) = coefficients;

    // Calculates the definite integral from 0 to upper_limit.
    let integral_upper = (a / 3.0) * upper_limit.powi(3)
        + (b / 2.0) * upper_limit.powi(2)
        + c * upper_limit;

    // Calculates the definite integral from 0 to lower_limit.
    let integral_lower = (a / 3.0) * lower_limit.powi(3)
        + (b / 2.0) * lower_limit.powi(2)
        + c * lower_limit;

    // Evaluates the definite integral from lower_limit to upper_limit.
    integral_upper - integral_lower
}
