using OpenCvSharp;
using System;

namespace CVCanny
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            var src = Cv2.ImRead("1.jpg", ImreadModes.AnyColor);
            var tmp = new Mat();
            Cv2.Canny(src, tmp, 90, 150);
            Cv2.ImShow("src", src);
            Cv2.ImShow("canny",tmp);
            Cv2.WaitKey();

        }
    }
}
