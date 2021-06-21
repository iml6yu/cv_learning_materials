using OpenCvSharp;
using System;

namespace CVTemplate
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            var src = Cv2.ImRead("1.png", ImreadModes.AnyColor);
            var temp = Cv2.ImRead("temp.png", ImreadModes.AnyColor);
            var size = temp.Size();
            Mat result = new Mat();
            Cv2.MatchTemplate(src, temp, result, TemplateMatchModes.SqDiffNormed);
            Point minLoc  ;
            Point maxLoc ;
            Point matchLoc = new Point(0, 0);
            Cv2.MinMaxLoc(result, out minLoc, out maxLoc);
            matchLoc = minLoc;
            Mat mask = src.Clone();
            //画框显示
            Cv2.Rectangle(mask, matchLoc, new Point(matchLoc.X + temp.Cols, matchLoc.Y + temp.Rows), Scalar.Green, 2);

            //新建窗体显示图片
            using (new Window("temp image", temp))
            using (new Window("src image", src))
            using (new Window("mask image", mask))
            {
                Cv2.WaitKey();
            }
        }
    }
}
