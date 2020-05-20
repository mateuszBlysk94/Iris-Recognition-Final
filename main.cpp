/*
 * Author: Mateusz Osiński, Warsaw University of Technology
 * This program is a part of master thesis: "Methods of iris detection in registered image".
 * The program downloads subsequent photos of the human eye from the disk location indicated in the code.
 * Using image filtering and processing methods, the program locates the iris of the eye. 
 * In the result image, the iris area is outlined by two circles and two segments.

 */

#include <opencv2/opencv.hpp>
 
using namespace std;
using namespace cv;


int main()
{
    // reading all images from folder, in for loop
    vector<cv::String> fn;
    glob("images/*.jpg", fn, false);
    vector<Mat> images;
    
    size_t count = fn.size(); //number of images
    for (size_t i=0; i<count; i++)
    {
        Mat src, pupilRed, resultImage;
        src = imread(fn[i]);
        resultImage = imread(fn[i]);
        Mat bgr[3];  
        split(src, bgr);

        //////////////// PUPIL DETECTING
        pupilRed=bgr[2];    //red layer
        
        threshold(pupilRed, pupilRed, 80, 255, 2);
        dilate(pupilRed, pupilRed, Mat());
        GaussianBlur(pupilRed, pupilRed, Size(17,17), 2, 0 );

        // circles detection - pupil
        vector<Vec3f> circlesPupil;
        HoughCircles(pupilRed, circlesPupil, CV_HOUGH_GRADIENT, 2, pupilRed.rows, 25, 15, 
                  pupilRed.rows/20, pupilRed.rows/6);

        // drawing a pupil circle on result image
        Point center(cvRound(circlesPupil[0][0]), cvRound(circlesPupil[0][1]));
        int radiusPupil = cvRound(circlesPupil[0][2]);
        circle(resultImage, center, 3, Scalar(0,0,255), -1, 8, 0 );
        circle(resultImage, center, radiusPupil, Scalar(0,0,255), 2, 8, 0 );
        
        
        //////////////// LIMBUS DETECTING
        
        cvtColor(src, src, CV_BGR2GRAY);
        // center of pupil
        int xp = circlesPupil[0][0]; //column
        int yp = circlesPupil[0][1]; //row

        // image of limbus edges
        Mat limbusEdgeImage (src.rows, src.cols, src.type());
        limbusEdgeImage = 0;
        int neighbours = 30; //number od pixels on the left and right 
        
        //this loop finds edges to the left of the pupil
        for (int i=neighbours; i<xp; i++)
        {
            for (int j=0; j<src.rows; j++)
            {
                float meanRight = 0;
                float meanLeft = 0;
                for (int k=0; k<neighbours; k++)
                {
                    //mean values od pixels on the right and left
                    meanRight += src.at<uchar>(j, i+k);
                    meanLeft += src.at<uchar>(j, i-k);
                }
                meanRight = meanRight / neighbours;
                meanLeft = meanLeft / neighbours;
                int actualValue = src.at<uchar>(j, i);
                if ( (abs(meanRight - actualValue) < 20 ) && (abs(meanLeft - actualValue) > 35) ) 
                    limbusEdgeImage.at<uchar>(j, i) = 255;
                else 
                    limbusEdgeImage.at<uchar>(j, i) = 0;
            }
        }
    
        //this loop finds edges to the right of the pupil
        for (int i=xp; i<src.cols-neighbours; i++)
        {
            for (int j=0; j<src.rows; j++)
            {
                float meanRight = 0;
                float meanLeft = 0;
                for (int k=0; k<neighbours; k++)
                {
                    meanRight += src.at<uchar>(j, i+k);
                    meanLeft += src.at<uchar>(j, i-k);
                }
                meanRight = meanRight / neighbours;
                meanLeft = meanLeft / neighbours;
                int actualValue = src.at<uchar>(j, i);
                if ( (abs(meanLeft - actualValue) < 20 ) && (abs(meanRight - actualValue) > 35) )
                    limbusEdgeImage.at<uchar>(j, i) = 255;
                else
                    limbusEdgeImage.at<uchar>(j, i) = 0;
            }
        }
    
        // deleting pixels in 30 - 150st and 225 - 315st
    
        // deleting on the left of the pupil
        for (int i=0; i<xp; i++)
        {
            for (int j=0; j<limbusEdgeImage.rows; j++)
            {
                float tangens = (float)(-j + yp) / (xp - i);
                if ((tangens<-1) || (tangens>0.577))
                    limbusEdgeImage.at<uchar>(j, i) = 0;
            }
        }
        
        // deleting on the right of the pupil
        for (int i=xp; i<limbusEdgeImage.cols; i++)
        {
            for (int j=0; j<limbusEdgeImage.rows; j++)
            {
                float tangens = (float)(-j + yp) / (xp - i);
                if ((tangens<-0.577) || (tangens>1))
                    limbusEdgeImage.at<uchar>(j, i) = 0;
            }
        }
    
        // circles detection - own function which reaplace the Hough Circle Transform method
        erode(limbusEdgeImage, limbusEdgeImage, Mat());  //deleting single pixels
    
        // vectors - table with informaction about detected circles
        vector <int> centerX, centerY, rLimb, power;
        
        // a loop which detects circles with center placed in pupil
        for (int r=(int)1.25 * radiusPupil; r<limbusEdgeImage.rows / 2; r+=4)
        {
            // an image of circles with centers in non-zero pixels
            Mat circles (limbusEdgeImage.rows, limbusEdgeImage.cols, limbusEdgeImage.type());
            circles = 0;
            
            // vectors with informaction about position of non-zero pixels
            vector <int> xFill;
            vector <int> yFill; 

            int iter = 0;
            for (int x=0; x<limbusEdgeImage.cols; x++)
            {
                for (int y=0; y<limbusEdgeImage.rows; y++)
                {
                    if (limbusEdgeImage.at<uchar>(y , x) != 0)
                    {
                        xFill.push_back(x);
                        yFill.push_back(y);
                        iter++;
                    }
                }
            }
       
            // random numbers
            int numRand = 80;
            int random[numRand] = {0};
            for (int i=0; i<numRand; i++)
                random[i] = rand() % iter;
        
            //drawing circles with centers in random non-zero pixels
            for (int i=0; i<numRand; i++)
            {
                Point center(xFill[random[i]], yFill[random[i]]);
                int radius = r;
                Mat temporary (limbusEdgeImage.rows, limbusEdgeImage.cols, limbusEdgeImage.type());
                temporary = 0;
                circle( temporary, center, radius, Scalar(1,0,0), 3, 8, 0 );
                circles += temporary;
            }
            //parameters of the best circle for actual radius r (from loop parameters)
            int actualValue = 0, centerx, centery;   
            for (int x=0; x<limbusEdgeImage.cols; x++)
            {
                for (int y=0; y<limbusEdgeImage.rows; y++)
                {
                    if (circles.at<uchar>(y , x) > actualValue)
                    {
                        centerx = x;
                        centery = y;
                        actualValue = circles.at<uchar>(y , x);
                    }
                }
            }
                
            // the circle is remembered if its center is inside the pupil
            if ( pow(xp - centerx, 2) + pow(yp - centery, 2) < pow(radiusPupil, 2))
            {
                centerX.push_back(centerx);
                centerY.push_back(centery);
                rLimb.push_back(r);
                power.push_back(actualValue);
            }
        }
    
        // choosing the best circle
        int maxPower = 0;
        int z = 0;
        for( size_t i = 0; i < power.size(); i++ )
        {
            if (power[i] > maxPower)
            {
                maxPower = power[i];
                z=i;
            }
        }

        // drawing limbus circle 
        Point centerLimb(centerX[z], centerY[z]);
        int radiusLimb = rLimb[z];
        circle(resultImage, centerLimb, radiusLimb, Scalar(255,0,0), 2, 8, 0 );
    
        //////////////// EYELIDS AREA - DRAWING LINES
        int x1, y1, x2, y2;
        x1 = sqrt(3)/2 * radiusLimb;
        y1 = radiusLimb / 2; //sqrt(1 - sqrt(3)/2);
        Point LH (centerX[z] - x1, centerY[z] - y1);
        Point RH (centerX[z] + x1, centerY[z] - y1);
        line(resultImage, LH, RH, Scalar(0, 255, 0), 2, 8, 0);
        
        x2 = sqrt(2)/2 * radiusLimb;
        y2 = radiusLimb * sqrt(2)/2;
        Point LL (centerX[z] - x2, centerY[z] + y2);
        Point RL (centerX[z] + x2, centerY[z] + y2);
        line(resultImage, LL, RL, Scalar(0, 255, 0), 2, 8, 0);
        
        
        // displaying the result image
        namedWindow("resultImage", WINDOW_AUTOSIZE);
        imshow("resultImage", resultImage);

        waitKey(0);
    }

    return 0;
}
