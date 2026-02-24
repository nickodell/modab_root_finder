// ModAB, from the original paper.
// Copyright ???


internal static double? ModAB(Func F, double x1, double x2, double eps)
{
    double y1 = F(x1), y2 = F(x2); //Evaluate the function at both ends
    int side = 0; //For tracking the side that has moved at the previous iteration
    int N = -(int)(Math.Log2(Precision) / 2) + 1; //Expected number of iterations
    double x0 = x1; //For storing the abscissa from the previous iteration
    bool Bisection = true;
    for (int i = 1; i <= n; ++i)
    {
        double x3, y3;
        if (Bisection) //Bisection step
        {
            x3 = (x1 + x2) / 2; y3 = F(x3); //Midpoint abscissa and function value
            double ym = (y1 + y2) / 2; //Ordinate of chord at midpoint
            if (Math.Abs(ym - y3) < 0.25 * (Math.Abs(ym) + Math.Abs(y3))
                Bisection = false; //Switch to false-position
        }
        else //False-position step
            x3 = (x1 * y2 – y1 * x2)/(y2 - y1); y3 = F(x3)
        if (y3 == 0 || Math.Abs(x3 - x0) <= eps) //Convergence check
            return x3; //Return the result
        x0 = x3; //Store the abscissa for the next iteration
        if (side == 1) //Apply Anderson-Bjork modification for side 1
        {
            double m = 1 - y3 / y1;
            if (m <= 0) y2 *= 0.5; else y2 *= m;
        }
        else if (side == 2) //Apply Anderson-Bjork modification for side 2
        {
            double m = 1 - y3 / y2;
            if (m <= 0) y1 /= 2; else y1 *= m;
        }
        if (Math.Sign(y1) == Math.Sign(y3)) //If the left interval does not change sign
        {
            if (!Bisection) side = 1; //Store the side that move
            x1 = x3; y1 = y3; //Move the left end
        }
        else //If the right interval does not change sign
        {
            if (!Bisection) side = 2; //Store the side that move
            x2 = x3; y2 = y3; //Move the right end
        }
        if (i % N == 0) //If the expected number of iterations is exceeded
        {
            Bisection = true; //Reset to bisection
            side = 0;
        }
    }
    return null; //If the algorithm failed to converge for n iterations
}
