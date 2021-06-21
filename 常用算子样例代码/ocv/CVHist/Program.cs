using OpenCvSharp;
using System;

namespace CVHist
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            Console.WriteLine("Hello World!");
            var src = Cv2.ImRead("1.png", ImreadModes.AnyColor);
            Mat gray = new Mat();

            Cv2.CvtColor(src, gray, ColorConversionCodes.BGR2GRAY);

            #region 自适应直方图均衡化
            Mat grayAdap = new Mat();
            var c = Cv2.CreateCLAHE(2, new Size(src.Width / 10, src.Height / 10));
            c.Apply(gray, grayAdap);
            Cv2.ImShow("grayAdap", grayAdap);
            Cv2.WaitKey();
            #endregion 自适应直方图均衡化


            #region 直方图均衡化
            Mat graynew = new Mat();
            Cv2.EqualizeHist(gray, graynew);
            Cv2.ImShow("graynew", graynew);
            Cv2.WaitKey();
            #endregion 直方图均衡化


            #region 掩膜  
            Mat mask = Mat.Zeros(src.Size(), MatType.CV_8UC1);

            for (var i = 200; i < 800; i++)
            {
                for (var j = 0; j < 600; j++)
                {
                    mask.Set(j, i, new Vec3b(255, 255, 255));
                }
            }
            Mat maskImg = new Mat();
            src.CopyTo(maskImg, mask);

            Cv2.ImShow("maskImg", maskImg);
            Cv2.WaitKey();

            #endregion 掩膜

            Mat hist = new Mat();
            Rangef[] range = new Rangef[1] { new Rangef(0.0f, 256.0f) };//一个通道，值范围 

            Cv2.CalcHist(new Mat[] { gray }, new int[] { 0 }, null, hist, 1, new int[] { 256 }, range);
            Console.WriteLine(hist.Rows + "行" + hist.Cols + "列");//把输出的行列打印出来

            Mat histImg = new Mat(500, 500, MatType.CV_8UC3);

            // 在前面的基础上
            //Console.WritleLine的后面加上
            for (int i = 0; i < 256; i++)//画直方图
            {
                // var a =  hist.Get<float>(i);

                int len = (int)((hist.Get<float>(i) / 10000) * histImg.Rows);//单个箱子的长度，
                                                                             // 10000) * panda.Rows)的作用只是让他变短，别超出了
                                                                             //Cv2.Line(histImg, i, 0, i, len, Scalar.Red, 2);//把线画出来
                Cv2.Line(histImg, i, 0, i, len, Scalar.White, 2);

            }

            Cv2.ImShow("hist", histImg);
            Cv2.WaitKey();
        }
    }
}
