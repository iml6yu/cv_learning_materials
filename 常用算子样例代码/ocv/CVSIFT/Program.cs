using OpenCvSharp;
using System;

namespace CVSIFT
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            var src = Cv2.ImRead("1.png", ImreadModes.AnyColor);
            Mat gray = new Mat();
            Cv2.CvtColor(src, gray, ColorConversionCodes.BGR2GRAY);

            //Mat hsv = new Mat();
            //Cv2.CvtColor(src, hsv, ColorConversionCodes.BGR2HSV);
            //Cv2.ImShow("hsv", hsv);
            //Cv2.WaitKey();
            //Mat hist = new Mat();
            //Rangef[] range = new Rangef[1] {  new Rangef(0.0f,256.0f)} ;//一个通道，值范围 

            //Cv2.CalcHist(new Mat[] { gray }, new int[] { 0 }, null, hist, 1, new int[] { 256 }, range);

            //Cv2.ImShow("hist", hist);
            //Cv2.WaitKey();
            using (var sift = OpenCvSharp.Features2D.SIFT.Create())
            {
                Mat mask = new Mat();
                Mat des = new Mat();
                KeyPoint[] keypoints;
                sift.DetectAndCompute(gray, mask, out keypoints, des);
                Cv2.DrawKeypoints(src, keypoints, src,null,DrawMatchesFlags.DrawRichKeypoints);
                Cv2.ImShow("src", src);
                Cv2.ImShow("gray", gray);
                Cv2.WaitKey(); 
            }

            using (var sift = OpenCvSharp.XFeatures2D.SURF.Create(100))
            {
                Mat mask = new Mat();
                Mat des = new Mat();
                KeyPoint[] keypoints;
                sift.DetectAndCompute(gray, mask, out keypoints, des);
                Cv2.DrawKeypoints(src, keypoints, src, null, DrawMatchesFlags.DrawRichKeypoints);
                Cv2.ImShow("src", src);
                Cv2.ImShow("gray", gray);
                Cv2.WaitKey();
            }

            //using (var sift = OpenCvSharp.XFeatures2D.SURF.Create(100))
            //{
            //    Mat mask = new Mat();
            //    Mat des = new Mat();
            //    KeyPoint[] keypoints;
            //    sift.DetectAndCompute(gray, mask, out keypoints, des);
            //    Cv2.DrawKeypoints(src, keypoints, src, null, DrawMatchesFlags.DrawRichKeypoints);
            //    Cv2.ImShow("src", src);
            //    Cv2.ImShow("gray", gray);
            //    Cv2.WaitKey();
            //}
        }
    }
}
