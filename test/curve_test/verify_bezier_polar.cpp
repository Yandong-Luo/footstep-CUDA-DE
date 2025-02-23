#include <bezier/bezier.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#define NUM_STEP 30

// Structure to hold polar coordinates
struct PolarPoint {
    double r;
    double theta;
};

// Convert Cartesian to Polar coordinates
PolarPoint cartesianToPolar(double x, double y) {
    double r = sqrt(x * x + y * y);
    double theta = atan2(y, x);
    return {r, theta};
}

// Convert Polar to Cartesian coordinates
bezier::Point polarToCartesian(double r, double theta) {
    return {r * cos(theta), r * sin(theta)};
}

// Function to generate control points in polar coordinates
std::vector<bezier::Point> generatePolarControlPoints(
    double x_start, double y_start, double vx_start, double vy_start, double theta_start,
    double x_end, double y_end, double vx_end, double vy_end, double theta_end
) {
    std::vector<bezier::Point> controlPoints;
    
    // Convert start and end points to polar coordinates
    PolarPoint start = cartesianToPolar(x_start, y_start);
    PolarPoint end = cartesianToPolar(x_end, y_end);
    
    // Convert velocities to polar coordinates
    PolarPoint v_start = cartesianToPolar(vx_start, vy_start);
    PolarPoint v_end = cartesianToPolar(vx_end, vy_end);
    
    // Start point
    controlPoints.push_back(polarToCartesian(start.r, start.theta));
    
    // First control point influenced by initial velocity
    double r1 = start.r + v_start.r/6.0;
    double theta1 = start.theta + theta_start/6.0;
    controlPoints.push_back(polarToCartesian(r1, theta1));
    
    // Generate intermediate control points
    for (int i = 2; i < 5; i++) {
        double t = i / 6.0;
        // Blend radii and angles
        double r_blend = (1 - t) * start.r + t * end.r;
        double theta_blend = (1 - t) * theta_start + t * theta_end;
        
        // Add some curvature influence
        double r_influence = 0.3 * (1 - pow(2*t-1, 2));
        r_blend += r_influence;
        
        controlPoints.push_back(polarToCartesian(r_blend, theta_blend));
    }
    
    // Last two control points influenced by final velocity and position
    double r5 = end.r - v_end.r/6.0;
    double theta5 = end.theta - theta_end/6.0;
    controlPoints.push_back(polarToCartesian(r5, theta5));
    controlPoints.push_back(polarToCartesian(end.r, end.theta));
    
    return controlPoints;
}

int main() {
    // Initial and final states
    double x_start = 0.29357406;
    double y_start = 0.29125562;
    double vx_start = -0.01193462;
    double vy_start = -0.01774755;
    double theta_start = 1.58432257;
    
    double x_end = 1.5;
    double y_end = 2.8;
    double vx_end = 0;
    double vy_end = 0;
    double theta_end = 0;
    
    // Generate control points using polar coordinates
    std::vector<bezier::Point> controlPoints = generatePolarControlPoints(
        x_start, y_start, vx_start, vy_start, theta_start,
        x_end, y_end, vx_end, vy_end, theta_end
    );
    
    // Create the 6th-order Bezier curve
    bezier::Bezier<6> sixthOrderBezier(controlPoints);
    
    // Create the first derivative curve for velocity calculation
    bezier::Bezier<5> firstDerivative = sixthOrderBezier.derivative();
    
    // Print control points in both Cartesian and Polar coordinates
    std::cout << "Control Points:" << std::endl;
    for (size_t i = 0; i < controlPoints.size(); i++) {
        PolarPoint polar = cartesianToPolar(controlPoints[i].x, controlPoints[i].y);
        std::cout << "P" << i << " - Cartesian: (" << controlPoints[i].x << ", " << controlPoints[i].y 
                 << "), Polar: (r=" << polar.r << ", θ=" << polar.theta << ")" << std::endl;
    }
    std::cout << std::endl;
    
    // Print points and velocities along the curve
    std::cout << std::setw(8) << "Step" 
              << std::setw(10) << "t" 
              << std::setw(12) << "X" 
              << std::setw(12) << "Y"
              << std::setw(12) << "r"
              << std::setw(12) << "θ"
              << std::setw(12) << "Vx" 
              << std::setw(12) << "Vy"
              << std::setw(12) << "Vr"
              << std::setw(12) << "Vθ"
              << std::setw(15) << "V_magnitude" 
              << std::endl;
    
    std::cout << std::string(120, '-') << std::endl;
    
    for (int i = 0; i <= NUM_STEP; i++) {
        double t = i / (double)NUM_STEP;
        
        // Position in Cartesian coordinates
        bezier::Point p = sixthOrderBezier.valueAt(t);
        
        // Convert to polar coordinates
        PolarPoint polar = cartesianToPolar(p.x, p.y);
        
        // Velocity in Cartesian coordinates
        bezier::Point v = firstDerivative.valueAt(t);
        
        // Convert velocity to polar coordinates
        // Vr = (x*vx + y*vy)/r
        // Vθ = (x*vy - y*vx)/r^2
        double vr = (p.x * v.x + p.y * v.y) / polar.r;
        double vtheta = (p.x * v.y - p.y * v.x) / (polar.r * polar.r);
        
        // Velocity magnitude
        double v_magnitude = sqrt(v.x * v.x + v.y * v.y);
        
        // Print formatted output
        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(8) << i
                  << std::setw(10) << t
                  << std::setw(12) << p.x
                  << std::setw(12) << p.y
                  << std::setw(12) << polar.r
                  << std::setw(12) << polar.theta
                  << std::setw(12) << v.x
                  << std::setw(12) << v.y
                  << std::setw(12) << vr
                  << std::setw(12) << vtheta
                  << std::setw(15) << v_magnitude
                  << std::endl;
    }
    
    // Print curve properties
    std::cout << "\nCurve Properties:" << std::endl;
    std::cout << "Curve length: " << sixthOrderBezier.length() << std::endl;
    
    // Calculate second derivative for acceleration
    bezier::Bezier<4> secondDerivative = firstDerivative.derivative();
    
    // Calculate maximum velocity and acceleration
    double maxVelocity = 0;
    double maxAcceleration = 0;
    
    for (int i = 0; i <= NUM_STEP; i++) {
        double t = i / (double)NUM_STEP;
        bezier::Point velocity = firstDerivative.valueAt(t);
        bezier::Point acceleration = secondDerivative.valueAt(t);
        
        double velocityMagnitude = sqrt(velocity.x * velocity.x + velocity.y * velocity.y);
        double accelerationMagnitude = sqrt(acceleration.x * acceleration.x + acceleration.y * acceleration.y);
        
        if (velocityMagnitude > maxVelocity) maxVelocity = velocityMagnitude;
        if (accelerationMagnitude > maxAcceleration) maxAcceleration = accelerationMagnitude;
    }
    
    std::cout << "Maximum velocity: " << maxVelocity << std::endl;
    std::cout << "Maximum acceleration: " << maxAcceleration << std::endl;
    
    return 0;
}