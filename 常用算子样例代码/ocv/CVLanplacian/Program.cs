using OpenCvSharp;
using System;

namespace CVLaplacian
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            var src = Cv2.ImRead("1.jpg", ImreadModes.AnyColor);
            Mat tmp = new Mat();
            Cv2.Laplacian(src, tmp, MatType.CV_16S);
            Mat result = new Mat();
            Cv2.ConvertScaleAbs(tmp, result);
            Cv2.ImShow("laplacian",result);
            Cv2.WaitKey();
        }
    }
}
