using OpenCvSharp;
using System;
using System.Drawing;

namespace ocv
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            Rect rect = new Rect(500, 100, 20, 40);

            Rectangle r = new Rectangle(500, 100, 100, 50);

            Mat mat = new Mat(10,100,MatType.CV_8UC3);
            mat.Rectangle(rect, Scalar.Blue);
            //Cv2.ImShow("test", mat);
            var size = mat.Size();
            Console.WriteLine(mat.Size(0));
            Console.WriteLine(mat.Size(1));
            Console.WriteLine(mat.Size(2));
            Console.WriteLine(mat.Step(0));
            Console.WriteLine(mat.Step(1)); 
            Console.WriteLine(mat.Step(2));
            VideoCapture cap
                 = new VideoCapture();
           
            //mat.FindNonZero 
            Console.WriteLine(mat.Depth());
            int channels = mat.Channels();                                              //获取mat 的通道数 用于底下的判断
            Console.WriteLine("channels={0}", channels);
            //for(var i = 0; i < mat.Rows; i++)
            //{
            //    for(var j = 0; j < mat.Cols; j++)
            //    {
            //        mat.Set(i, j, new Vec3b((byte)(i %255), (byte)(j % 255), 0));
            //    }
            //}
          
          
            //mat.Set(10, 10, new Vec3b(255, 255, 255));
            //mat.SetArray<Vec3b>(new Vec3b[] { new Vec3b(255,255,255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255), new Vec3b(255, 255, 255) });
            //var mat1 = new Mat(mat, rect);
            using (new Window("out", mat))
            //using (new Window("out", mat1))
            {
                Cv2.WaitKey();
            }
        }
    }
}
