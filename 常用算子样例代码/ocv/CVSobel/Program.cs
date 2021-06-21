using OpenCvSharp;
using System;

namespace CVSobel
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            // Mat src = new Mat("1.jpg");

            var gray = Cv2.ImRead("1.jpg",ImreadModes.Grayscale);
            Mat xgrad = new Mat();
            Mat ygrad = new Mat();
            Cv2.Sobel(gray, xgrad, MatType.CV_16S, 1, 0, -1);
            Cv2.Sobel(gray, ygrad, MatType.CV_16S, 0, 1,-1);
            Cv2.ConvertScaleAbs(xgrad, xgrad);//缩放、计算绝对值并将结果转换为8位。不做转换的化显示不了，显示图相只能是8U类型 
            Cv2.ConvertScaleAbs(ygrad, ygrad);
            Mat output = new Mat(xgrad.Size(), xgrad.Type());
            Cv2.AddWeighted(xgrad, 0.5, ygrad, 0.5, 1, output);
            Cv2.ImShow("0-1", output);

            Cv2.WaitKey();

        }
    }
}
