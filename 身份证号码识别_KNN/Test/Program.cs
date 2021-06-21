using OpenCvSharp;
using OpenCvSharp.ML;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection.Metadata.Ecma335;
using System.Security.Claims;

namespace Test
{
    class Program
    {
        static void Main(string[] args)
        {
            ////KNNNumberTrain();
            // KNNTrain();

            //var fs = System.IO.Directory.GetFiles(@"C:\liuyu\LY.OpenCVSharpLib\images\test");
            //foreach(var f in fs)
            //{
            //   var rr= KNNClassfilter(f);
            //    if (!System.IO.Directory.Exists(@"C:\liuyu\LY.OpenCVSharpLib\images\test\\"+rr))
            //        System.IO.Directory.CreateDirectory(@"C:\liuyu\LY.OpenCVSharpLib\images\test\\" + rr);
            //    File.Copy(f, @"C:\liuyu\LY.OpenCVSharpLib\images\test\\" + rr+"\\"+Guid.NewGuid()+".jpg");
            //    Console.WriteLine(rr);
            //}

            // Console.ReadLine();


            Console.WriteLine("Hello World!");
            Mat src = Cv2.ImRead(@"C:\liuyu\LY.OpenCVSharpLib\images\idcard.jpg");
            Mat resizeSrc = new Mat();
            Cv2.Resize(src, resizeSrc, new Size(1000, 650));


            Mat gray = new Mat();
            Cv2.CvtColor(resizeSrc, gray, ColorConversionCodes.BGR2GRAY);
            //Cv2.ImShow("gray", gray);
            //Cv2.WaitKey();

            //Mat disGray = new Mat();
            //Cv2.BitwiseNot(gray, disGray);
            //Cv2.ImShow("disGray", disGray);
            //Cv2.WaitKey();

            Mat threshold = new Mat();
            Cv2.Threshold(gray, threshold, 100, 255, ThresholdTypes.BinaryInv);
            //Cv2.ImShow("threshold", threshold);
            //Cv2.WaitKey();
            //膨胀操作
            Mat dilation = new Mat();
            var ele = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(20, 15));
            Cv2.Dilate(threshold, dilation, ele, iterations: 1);
            //Cv2.ImShow("dil", dilation);
            //Cv2.WaitKey();

            Point[][] contouts;
            HierarchyIndex[] hies;
            Cv2.FindContours(dilation, out contouts, out hies, RetrievalModes.Tree, ContourApproximationModes.ApproxSimple);
            Point[][] targetContouts = new Point[contouts.Length][];
            int index = 0;
            foreach (var points in contouts)
            {
                var topleft = new Point(points.Min(t => t.X), points.Min(t => t.Y));
                var topright = new Point(points.Max(t => t.X), points.Min(t => t.Y));
                var bottomleft = new Point(points.Min(t => t.X), points.Max(t => t.Y));
                var bottomright = new Point(points.Max(t => t.X), points.Max(t => t.Y));
                targetContouts[index] = new Point[4];
                targetContouts[index][0] = topleft;
                targetContouts[index][1] = topright;
                targetContouts[index][2] = bottomright;
                targetContouts[index][3] = bottomleft;
                index++;
            }
            List<Point[]> r = new List<Point[]>();
            foreach (var item in AAA(targetContouts))
            {
                r.Add(item.ToArray());
            }
            Cv2.Polylines(resizeSrc, r.ToArray(), true, Scalar.Red);
            //Cv2.ImShow("Polylines", resizeSrc);
            var codePloy = GetCarNumberPosition(r, resizeSrc);
            if (codePloy != null)
            {
                Mat code = new Mat(resizeSrc, (Rect)codePloy);
                //Cv2.ImShow("code", code);
                //Cv2.WaitKey();
                var codeGray = new Mat();
                Cv2.CvtColor(code, codeGray, ColorConversionCodes.BGR2GRAY);
                var codeThreshold = new Mat();
                Cv2.Threshold(codeGray, codeThreshold, 100, 255, ThresholdTypes.BinaryInv);



                ele = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(2, 2));
                var gus = new Mat();
                Cv2.MorphologyEx(codeThreshold, gus, MorphTypes.Close, ele);

                Mat codeDilation = new Mat();
                ele = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(6, 6));
                Cv2.Dilate(gus, codeDilation, ele, iterations: 1);
                //Cv2.ImShow("gus", codeDilation);
                //KNNTrain();
                // KNNClassfilter();


                Cv2.FindContours(codeDilation, out contouts, out hies, RetrievalModes.External, ContourApproximationModes.ApproxSimple);
                var rects = contouts.Select(t => Cv2.BoundingRect(t)).Where(t => t.Width / t.Height < 2).OrderBy(t => t.X).ToList();
                var result = "";
                // index = 0;
                rects.ForEach(rect =>
                {
                    Cv2.Rectangle(code, rect, Scalar.Red);

                    using (Mat temp = new Mat(gus, rect))
                    {
                        //temp.ImWrite(@"C:\liuyu\LY.OpenCVSharpLib\images\" + $"{index}.jpg");

                        //Cv2.WaitKey();
                        ////  
                        var rr = KNNClassfilter(temp);
                        if (rr == "10")
                            rr = "x";
                        Console.WriteLine(rr);
                        result += rr;
                    }
                    // index++;

                });
                Console.WriteLine(result);
                Console.ReadLine();

                //Cv2.ImShow("code", code);
                //Mat codeDilation = new Mat();
                //ele = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(10,10));
                //Cv2.Dilate(codeThreshold, codeDilation, ele, iterations: 1);
                //Cv2.ImShow("threshold", codeDilation);

                //Cv2.FindContours(codeDilation, out contouts, out hies, RetrievalModes.Tree, ContourApproximationModes.ApproxSimple);
                //Cv2.Polylines(code, contouts, false, Scalar.Red);
                //Cv2.ImShow("code", code);
            }




            Cv2.WaitKey();
        }

        public static System.Collections.Generic.IEnumerable<System.Collections.Generic.IEnumerable<Point>> AAA(Point[][] targetContouts)
        {
            for (var i = 0; i < targetContouts.Length; i++)
            {
                var ps = targetContouts[i];
                if (Cv2.ContourArea(ps) > 1000)
                {
                    yield return ps;
                }
            }
        }
        public static Rect? GetCarNumberPosition(List<Point[]> targetContouts, Mat src)
        {
            foreach (var p in targetContouts)
            {
                if (p[0].X > src.Rows / 2 && (p[2].X - p[0].X) / (p[2].Y - p[0].Y) > 6)
                    return new Rect(p[0], new Size(p[2].X - p[0].X, p[2].Y - p[0].Y));
            }
            return null;
        }

        public static void KNNTrain()
        {
            var files = System.IO.Directory.GetFiles(@"C:\liuyu\LY.OpenCVSharpLib\images\1\").OrderBy(t => t);
            Mat labels = new Mat(1, 11, MatType.CV_32S, new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
            List<float> buffer = new List<float>();
            List<Mat> mats = new List<Mat>();

            foreach (var file in files)
            {
                var mat = Cv2.ImRead(file, ImreadModes.Grayscale);
                mats.Add(mat);
                var resizemat = new Mat();
                Cv2.Resize(mat, resizemat, new Size(37, 26));
                Console.WriteLine(resizemat.Reshape(1, 1).Cols);
                var m = new Mat();
                resizemat.ConvertTo(m, MatType.CV_32FC1);
                //  _0Features = _0Features.Reshape(1, 1);
                float[] a;

                m.GetArray(out a);
                buffer.AddRange(a);

                //var resize = new Mat();
                //Cv2.Resize(mat, resize, new Size(600, 600));
                //Console.WriteLine(resize.ToBytes().Length);
                //buffer.AddRange(resize.ToBytes());
                ////var bytes = string.Join(",", mat.ToBytes());
            }

            //var maxMat = new Mat(new Size(mats.Max(t => t.Rows), mats.Max(t => t.Cols)), 0);
            var max = mats.Max(t => t.Rows) * mats.Max(t => t.Cols);
            //mats.ForEach(mat =>
            //{

            //    ////var bytes = mat.ToBytes();
            //    //byte[] bs = new byte[max - bytes.Count()];
            //    //var newbytes = bytes.Concat(bs);
            //    //buffer.AddRange(newbytes);
            //});

            Mat samples = new Mat(11, max, MatType.CV_32FC1, buffer.ToArray());

            using (var knn = KNearest.Create())
            {
                knn.Train(samples, SampleTypes.RowSample, labels);
                knn.Save(@"C:\liuyu\LY.OpenCVSharpLib\images\numall.yml");
            }

        }

        public static string KNNClassfilter(string filename)
        {
            using (var knn = KNearest.Load(@"C:\liuyu\LY.OpenCVSharpLib\images\numall.yml"))
            {
                Mat s = Cv2.ImRead(filename, 0);
                Mat resize = new Mat();
                Cv2.Resize(s, resize, new Size(37, 26));
                Mat f = new Mat();
                resize.ConvertTo(f, MatType.CV_32FC1);
                f = f.Reshape(1, 1);
                Mat result = new Mat();

                var neighborResponses = new Mat();
                var dists = new Mat();
                var r = knn.FindNearest(f, 1, result, neighborResponses, dists);
                var t = r.ToString(CultureInfo.InvariantCulture);
                return t;
            }
        }

        public static string KNNClassfilter(Mat s)
        {
            using (var knn = KNearest.Load(@"C:\liuyu\LY.OpenCVSharpLib\images\numall.yml"))
            {

                Mat resize = new Mat();
                Cv2.Resize(s, resize, new Size(37, 26));
                Mat f = new Mat();
                resize.ConvertTo(f, MatType.CV_32FC1);
                f = f.Reshape(1, 1);
                Mat result = new Mat();

                var neighborResponses = new Mat();
                var dists = new Mat();
                var r = knn.FindNearest(f, 1, result, neighborResponses, dists);
                var t = r.ToString(CultureInfo.InvariantCulture);
                return t;
            }
        }
        /// <summary>
        /// Mat 转成另外一种存储矩阵方式
        /// </summary>
        /// <param name="roi"></param>
        /// <returns></returns>
        public static Mat ConvertFloat(Mat roi)
        {
            var resizedImage = new Mat();
            var resizedImageFloat = new Mat();
            Cv2.Resize(roi, resizedImage, new Size(37, 26)); //resize to 10X10
            resizedImage.ConvertTo(resizedImageFloat, MatType.CV_32FC1); //convert to float
            var result = resizedImageFloat.Reshape(1, 1);
            return result;
        }


        public static void KNNNumberTrain()
        {
            float[] trainFeaturesData =
            {
                 2,2,2,2,
                 3,3,3,3,
                 4,4,4,4,
                 5,5,5,5,
                 6,6,6,6,
                 7,7,7,7
            };
            var trainFeatures = new Mat(6, 4, MatType.CV_32F, trainFeaturesData);

            int[] trainLabelsData = { 2, 3, 4, 5, 6, 7 };
            var trainLabels = new Mat(1, 6, MatType.CV_32S, trainLabelsData);

            Mat _0 = Cv2.ImRead(@"C:\liuyu\LY.OpenCVSharpLib\images\1\0.jpg", 0);
            Mat _0Features = new Mat();
            _0.ConvertTo(_0Features, MatType.CV_32FC1);
            _0Features = _0Features.Reshape(1, 1);



            var _0Label = new Mat(1, 1, MatType.CV_32S, new int[] { 0 });

            using var kNearest = KNearest.Create();
            kNearest.Train(_0Features, SampleTypes.RowSample, _0Label);
            kNearest.Save(@"C:\liuyu\LY.OpenCVSharpLib\images\num0.yml");
            float[] testFeatureData = new float[] {
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0,
  0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
  1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2,
  2, 2, 0, 0, 0, 0, 0, 0, 4, 4, 4, 1, 1, 1, 2, 2,
  2, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0,
  0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0,
  1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
  0, 0, 2, 2, 2, 0, 0, 0, 250, 250, 250, 5, 5, 5,
  0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0,
  0, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 3, 3, 3, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2,
  1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0,
  0, 1, 1, 1, 0, 0, 0, 3, 3, 3, 2, 2, 2, 0, 0, 0,
  251, 251, 251, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 251, 251, 251, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 4, 4, 4, 6, 6, 6, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
  0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 251, 251, 251, 0, 0, 0, 0, 0,
  0, 2, 2, 2, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 1,
  1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
  0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 1, 1, 1, 0, 0, 0, 2, 2, 2, 0, 0, 0, 4, 4,
  4, 0, 0, 0, 2, 2, 2, 251, 251, 251, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 1,
  1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0,
  0, 0, 251, 251, 251, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
  1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2,
  2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 251, 251,
  251, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2,
  2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
  1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  4, 4, 4, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1,
  1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 1, 1, 1, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 251, 251, 251, 0, 0,
  0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 1, 1, 1, 0,
  0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2,
  1, 1, 1, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3,
  3, 3, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 2, 2, 2, 4, 4, 4, 0, 0, 0, 2,
  2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 1, 1, 1, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 1, 1, 1, 3, 3,
  3, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 4,
  4, 4, 0, 0, 0, 3, 3, 3, 0, 0, 0, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 249, 249, 249, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1,
  0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 251, 251,
  251, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
  2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
  2, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
  0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0, 0,
  2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0,
  0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
  1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 0,
  0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0,
  0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 2, 2, 2,
  0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
  0, 0, 0, 2, 2, 2, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
  0, 0, 3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0, 2, 2,
  2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
  2, 2, 2, 2, 2, 3, 3, 3, 0, 0, 0, 2, 2, 2, 0, 0,
  0, 0, 0, 0, 1, 1, 1, 0, 0, 0
            };
            var testFeature = new Mat(1, testFeatureData.Length, MatType.CV_32F, testFeatureData);

            const int k = 1;
            var results = new Mat();
            var neighborResponses = new Mat();
            var dists = new Mat();
            try
            {
                var detectedClass = (int)kNearest.FindNearest(testFeature, 1, results, neighborResponses, dists);

                Console.WriteLine(detectedClass);
            }
            catch (OpenCvSharp.OpenCVException ex)
            {
                Console.WriteLine("Not found");
            }

        }
    }
}
