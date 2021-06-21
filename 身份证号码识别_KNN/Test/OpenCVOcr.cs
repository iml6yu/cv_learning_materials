using OpenCvSharp;
using OpenCvSharp.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Test
{
    public class OpencvOcr
    {
        #region Constructor

        const double Thresh = 80;
        const double ThresholdMaxVal = 255;
        const int _minHeight = 35;
        bool _isDebug = false;
        KNearest _KNearest = null;
        OpencvOcrConfig _config = new OpencvOcrConfig() { ZoomLevel = 2, ErodeLevel = 3 };
        #endregion

        /// <summary>
        /// 构造函数
        /// </summary>
        /// <param name="path">训练库完整路径</param>
        /// <param name="opencvOcrConfig">OCR相关配置信息</param>
        public OpencvOcr(string path, OpencvOcrConfig opencvOcrConfig = null)
        {
            if (string.IsNullOrEmpty(path))
                throw new ArgumentNullException("path is not null");

            if (opencvOcrConfig != null)
                _config = opencvOcrConfig;

            this.LoadKnearest(path);
        }

        /// <summary>
        /// 加载Knn 训练库模型
        /// </summary>
        /// <param name="dataPathFile"></param>
        /// <returns></returns>
        private KNearest LoadKnearest(string dataPathFile)
        {
            if (_KNearest == null)
            {

                using (var fs = new FileStorage(dataPathFile, FileStorage.Modes.Read))
                {
                    var samples = fs["samples"].ReadMat();
                    var responses = fs["responses"].ReadMat();
                    this._KNearest = KNearest.Create();
                    this._KNearest.Train(samples, SampleTypes.RowSample, responses);
                }
            }
            return _KNearest;
        }

        /// <summary>
        /// OCR 识别,仅仅只能识别单行数字 
        /// </summary>
        /// <param name="kNearest">训练库</param>
        /// <param name="path">要识别的图片路径</param>
        public override string GetText(Mat src, bool isDebug = false)
        {
            this._isDebug = isDebug;

            #region 图片处理
            var respMat = MatProcessing(src, isDebug);
            if (respMat == null)
                return "";
            #endregion

            #region 查找轮廓
            var sortRect = FindContours(respMat.FindContoursMat);
            #endregion

            return GetText(sortRect, respMat.ResourcMat, respMat.RoiResultMat);
        }

        /// <summary>
        /// 查找轮廓
        /// </summary>
        /// <param name="src"></param>
        /// <returns></returns>
        private List<Rect> FindContours(Mat src)
        {
            try
            {
                #region 查找轮廓
                Point[][] contours;
                HierarchyIndex[] hierarchyIndexes;
                Cv2.FindContours(
                    src,
                    out contours,
                    out hierarchyIndexes,
                    mode: OpenCvSharp.ContourRetrieval.External,
                    method: OpenCvSharp.ContourChain.ApproxSimple);

                if (contours.Length == 0)
                    throw new NotSupportedException("Couldn't find any object in the image.");
                #endregion

                #region 单行排序（目前仅仅支持单行文字,多行文字顺序可能不对，按照x坐标进行排序）
                var sortRect = GetSortRect(contours, hierarchyIndexes);
                sortRect = sortRect.OrderBy(item => item.X).ToList();
                #endregion

                return sortRect;
            }
            catch { }

            return null;
        }

        /// <summary>
        /// 获得切割后的数量列表
        /// </summary>
        /// <param name="contours"></param>
        /// <param name="hierarchyIndex"></param>
        /// <returns></returns>
        private List<Rect> GetSortRect(Point[][] contours, HierarchyIndex[] hierarchyIndex)
        {
            var sortRect = new List<Rect>();

            var _contourIndex = 0;
            while ((_contourIndex >= 0))
            {
                var contour = contours[_contourIndex];
                var boundingRect = Cv2.BoundingRect(contour); //Find bounding rect for each contour

                sortRect.Add(boundingRect);
                _contourIndex = hierarchyIndex[_contourIndex].Next;
            }
            return sortRect;
        }


        /// <summary>
        /// 是否放大
        /// </summary>
        /// <param name="src"></param>
        /// <returns></returns>
        private bool IsZoom(Mat src)
        {
            if (src.Height <= _minHeight)
                return true;

            return false;
        }


        private List<EnumMatAlgorithmType> GetAlgoritmList(Mat src)
        {
            var result = new List<EnumMatAlgorithmType>();
            var algorithm = this._config.Algorithm;

            #region 自定义的算法
            try
            {
                if (algorithm.Contains("|"))
                {
                    result = algorithm.Split('|').ToList()
                        .Select(item => (EnumMatAlgorithmType)Convert.ToInt32(item))
                        .ToList();

                    if (!IsZoom(src))
                        result.Remove(EnumMatAlgorithmType.Zoom);

                    return result;
                }
            }
            catch { }

            #endregion

            #region 默认算法
            if (IsZoom(src))
            {
                result.Add(EnumMatAlgorithmType.Zoom);
            }
            if (this._config.ThresholdType == ThresholdType.Binary)
            {
                //result.Add(EnumMatAlgorithmType.Blur);

                result.Add(EnumMatAlgorithmType.Gray);
                result.Add(EnumMatAlgorithmType.Thresh);
                if (this._config.DilateLevel > 0)
                    result.Add(EnumMatAlgorithmType.Dilate);

                result.Add(EnumMatAlgorithmType.Erode);
                return result;
            }
            //result.Add(EnumMatAlgorithmType.Blur);

            result.Add(EnumMatAlgorithmType.Gray);
            result.Add(EnumMatAlgorithmType.Thresh);
            if (this._config.DilateLevel > 0)
                result.Add(EnumMatAlgorithmType.Dilate);

            result.Add(EnumMatAlgorithmType.Erode);
            return result;
            #endregion
        }


        /// <summary>
        /// 对查找的轮廓数据进行训练模型匹配，这里使用的是KNN 匹配算法
        /// </summary>
        private string GetText(List<Rect> sortRect, Mat source, Mat roiSource)
        {
            var response = "";
            try
            {
                if ((sortRect?.Count ?? 0) <= 0)
                    return response;

                var contourIndex = 0;
                using (var dst = new Mat(source.Rows, source.Cols, MatType.CV_8UC3, Scalar.All(0)))
                {
                    sortRect.ForEach(boundingRect =>
                    {
                        try
                        {
                            #region 绘制矩形
                            if (this._isDebug)
                            {
                                Cv2.Rectangle(source, new Point(boundingRect.X, boundingRect.Y),
                                new Point(boundingRect.X + boundingRect.Width, boundingRect.Y + boundingRect.Height),
                                new Scalar(0, 0, 255), 1);

                                Cv2.Rectangle(roiSource, new Point(boundingRect.X, boundingRect.Y),
                                   new Point(boundingRect.X + boundingRect.Width, boundingRect.Y + boundingRect.Height),
                                   new Scalar(0, 0, 255), 1);
                            }
                            #endregion

                            #region 单个ROI
                            var roi = roiSource.GetROI(boundingRect); //Crop the image
                            roi = roi.Compress();
                            var result = roi.ConvertFloat();
                            #endregion

                            #region KNN 匹配
                            var results = new Mat();
                            var neighborResponses = new Mat();
                            var dists = new Mat();
                            var detectedClass = (int)this._KNearest.FindNearest(result, 1, results, neighborResponses, dists);
                            var resultText = detectedClass.ToString(CultureInfo.InvariantCulture);
                            #endregion

                            #region 匹配
                            var isDraw = false;
                            if (detectedClass >= 0)
                            {
                                response += detectedClass.ToString();
                                isDraw = true;
                            }
                            if (detectedClass == -1 && !response.Contains("."))
                            {
                                response += ".";
                                resultText = ".";
                                isDraw = true;
                            }
                            #endregion

                            #region 绘制及输出切割信息库
                            try
                            {
                                //if (this._isDebug)
                                //{
                                Write(contourIndex, detectedClass, roi);
                                //}
                            }
                            catch { }

                            if (this._isDebug && isDraw)
                            {
                                Cv2.PutText(dst, resultText, new Point(boundingRect.X, boundingRect.Y + boundingRect.Height), 0, 1, new Scalar(0, 255, 0), 2);
                            }
                            #endregion

                            result?.Dispose();
                            results?.Dispose();
                            neighborResponses?.Dispose();
                            dists?.Dispose();
                            contourIndex++;
                        }
                        catch (Exception ex)
                        {
                            TextHelper.Error("GetText ex", ex);
                        }
                    });

                    #region 调试模式显示过程
                    source.IsDebugShow("Segmented Source", this._isDebug);
                    dst.IsDebugShow("Detected", this._isDebug);
                    dst.IsDebugWaitKey(this._isDebug);
                    dst.IsDebugImWrite("dest.jpg", this._isDebug);
                    #endregion
                }
            }
            catch
            {
                throw;
            }
            finally
            {
                source?.Dispose();
                roiSource?.Dispose();
            }
            return response;
        }

        /// <summary>
        /// 图片处理算法
        /// </summary>
        /// <param name="src"></param>
        /// <param name="isDebug"></param>
        /// <returns></returns>
        public ImageProcessModel MatProcessing(Mat src, bool isDebug = false)
        {
            src.IsDebugShow("原图", isDebug);

            var list = GetAlgoritmList(src);
            var resultMat = new Mat();
            src.CopyTo(resultMat);
            var isZoom = IsZoom(src);
            list?.ForEach(item =>
            {
                switch (item)
                {
                    case EnumMatAlgorithmType.Dilate:
                        resultMat = resultMat.ToDilate(Convert.ToInt32(this._config.DilateLevel));
                        resultMat.IsDebugShow(EnumMatAlgorithmType.Dilate.GetDescription(), isDebug);
                        break;
                    case EnumMatAlgorithmType.Erode:
                        var eroderLevel = isZoom ? this._config.ErodeLevel * this._config.ZoomLevel : this._config.ErodeLevel;
                        resultMat = resultMat.ToErode(eroderLevel);
                        resultMat.IsDebugShow(EnumMatAlgorithmType.Erode.GetDescription(), isDebug);
                        break;
                    case EnumMatAlgorithmType.Gray:
                        resultMat = resultMat.ToGrey();
                        resultMat.IsDebugShow(EnumMatAlgorithmType.Gray.GetDescription(), isDebug);
                        break;
                    case EnumMatAlgorithmType.Thresh:
                        var thresholdValue = this._config.ThresholdValue <= 0 ? resultMat.GetMeanThreshold() : this._config.ThresholdValue;
                        resultMat = resultMat.ToThreshold(thresholdValue, thresholdType: this._config.ThresholdType);
                        resultMat.IsDebugShow(EnumMatAlgorithmType.Thresh.GetDescription(), isDebug);
                        break;
                    case EnumMatAlgorithmType.Zoom:
                        resultMat = resultMat.ToZoom(this._config.ZoomLevel);
                        src = resultMat;
                        resultMat.IsDebugShow(EnumMatAlgorithmType.Zoom.GetDescription(), isDebug);
                        break;
                    case EnumMatAlgorithmType.Blur:
                        resultMat = resultMat.ToBlur();
                        src = resultMat;
                        resultMat.IsDebugShow(EnumMatAlgorithmType.Blur.GetDescription(), isDebug);
                        break;
                }
            });

            var oldThreshImage = new Mat();
            resultMat.CopyTo(oldThreshImage);

            return new ImageProcessModel()
            {
                ResourcMat = src,
                FindContoursMat = oldThreshImage,
                RoiResultMat = resultMat
            };
        }
    }

    public class OpencvOcrConfig
    {
        /// <summary>
        /// 放大程度级别 默认2
        /// </summary>
        public double ZoomLevel { set; get; }

        /// <summary>
        /// 腐蚀级别 默认2.5
        /// </summary>
        public double ErodeLevel { set; get; }

        /// <summary>
        /// 膨胀
        /// </summary>
        public double DilateLevel { set; get; }

        /// <summary>
        /// 阀值
        /// </summary>
        public double ThresholdValue { set; get; }

        /// <summary>
        /// 图片处理算法,用逗号隔开
        /// </summary>
        public string Algorithm { set; get; }

        /// <summary>
        /// 二值化方式
        /// </summary>
        public ThresholdType ThresholdType { set; get; } = ThresholdType.BinaryInv;

        /// <summary>
        /// 通道模式
        /// </summary>
        public OcrChannelTypeEnums ChannelType { set; get; } = OcrChannelTypeEnums.BlackBox;

    }
}
