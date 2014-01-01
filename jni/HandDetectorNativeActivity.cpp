/*
   HandDetector for Android NDK
   Copyright (c) 2006-2013 SIProp Project http://www.siprop.org/

   This software is provided 'as-is', without any express or implied warranty.
   In no event will the authors be held liable for any damages arising from the use of this software.
   Permission is granted to anyone to use this software for any purpose,
   including commercial applications, and to alter it and redistribute it freely,
   subject to the following restrictions:

   1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
   2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
   3. This notice may not be removed or altered from any source distribution.
*/
#include "android_native_app_glue.h"
#include <android/log.h>
#include <omp.h>

#include <errno.h>
#include <sys/time.h>
#include <time.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <queue>
#include <string>

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/cvaux.h>
#include <opencv/ml.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <assetmanager.h>

#include "HandDetector.hpp"

#define  LOG_TAG    "NativeActivity"
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG,__VA_ARGS__)
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGW(...)  __android_log_print(ANDROID_LOG_WARN,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)


#define DATA_PATH "/data/data"
#define PACKAGE_NAME "org.siprop.android.handdetectoropenmp"

#define WIDTH 640
#define HEIGHT 480


#define BENCHMARK

#define REPEATE 10


struct Engine {
	struct android_app* app;
#ifndef BENCHMARK
    cv::Ptr<cv::VideoCapture> capture;
#endif
};

int createAssetFile(const char *image_name) {
	const char package_name[] = PACKAGE_NAME;
	int result = 0;
	result = loadAseetFile(package_name, image_name);
	return result;
}

#ifdef BENCHMARK
static void engine_draw_frame(Engine* engine, const cv::Mat& rot_frame) {
#else
static void engine_draw_frame(Engine* engine, const cv::Mat& frame) {
#endif

    if (engine->app->window == NULL)
        return; // No window.

#ifndef BENCHMARK
    cv::Mat tmp_frame(cv::Size(HEIGHT, WIDTH), frame.type(), cv::Scalar(0, 0, 0));
    cv::Mat rot_frame(cv::Size(WIDTH, HEIGHT), frame.type());
    cv::transpose(frame, tmp_frame);
    cv::flip(tmp_frame, tmp_frame, 1);
    cv::resize(tmp_frame, rot_frame, rot_frame.size(), cv::INTER_CUBIC);
    LOGI("Finish rotate frame.");
#endif

    ANativeWindow_Buffer buffer;
    if (ANativeWindow_lock(engine->app->window, &buffer, NULL) < 0) {
        LOGW("Unable to lock window buffer");
        return;
    }

    int32_t* pixels = (int32_t*)buffer.bits;

    int left_indent = (buffer.width-rot_frame.cols)/2;
    int top_indent = (buffer.height-rot_frame.rows)/2;

    if (top_indent > 0) {
        memset(pixels, 0, top_indent*buffer.stride*sizeof(int32_t));
        pixels += top_indent*buffer.stride;
    }

    for (int yy = 0; yy < rot_frame.rows; yy++) {
        if (left_indent > 0){
            memset(pixels, 0, left_indent*sizeof(int32_t));
            memset(pixels+left_indent+rot_frame.cols, 0, (buffer.stride-rot_frame.cols-left_indent)*sizeof(int32_t));
        }
        int32_t* line = pixels + left_indent;
        size_t line_size = rot_frame.cols*4*sizeof(unsigned char);
        memcpy(line, rot_frame.ptr<unsigned char>(yy), line_size);
        // go to next line
        pixels += buffer.stride;
    }
    ANativeWindow_unlockAndPost(engine->app->window);
}

static void engine_handle_cmd(android_app* app, int32_t cmd) {
    Engine* engine = (Engine*)app->userData;
    switch (cmd) {
        case APP_CMD_START:
        	LOGI("APP_CMD_START");
        	break;
        case APP_CMD_RESUME:
        	LOGI("APP_CMD_RESUME");
        	break;
        case APP_CMD_SAVE_STATE:
        	LOGI("APP_CMD_SAVE_STATE");
        	break;
        case APP_CMD_PAUSE:
        	LOGI("APP_CMD_PAUSE");
        	break;
        case APP_CMD_STOP:
        	LOGI("APP_CMD_STOP");
        	break;
        case APP_CMD_DESTROY:
        	LOGI("APP_CMD_DESTROY");
        	break;
        case APP_CMD_GAINED_FOCUS:
        	LOGI("APP_CMD_GAINED_FOCUS");
        	break;
        case APP_CMD_LOST_FOCUS:
        	LOGI("APP_CMD_LOST_FOCUS");
        	break;
        case APP_CMD_INIT_WINDOW:
        	LOGI("APP_CMD_INIT_WINDOW");
            if (app->window != NULL) {

                int view_width = WIDTH;
                int view_height = HEIGHT;
#ifndef BENCHMARK
                engine->capture = new cv::VideoCapture(0);

                union {double prop; const char* name;} u;
                u.prop = engine->capture->get(CV_CAP_PROP_SUPPORTED_PREVIEW_SIZES_STRING);

                cv::Size camera_resolution = cv::Size(WIDTH, HEIGHT);
                if ((camera_resolution.width != 0) && (camera_resolution.height != 0)) {
                    engine->capture->set(CV_CAP_PROP_FRAME_WIDTH, camera_resolution.width);
                    engine->capture->set(CV_CAP_PROP_FRAME_HEIGHT, camera_resolution.height);
                }
                LOGI("Camera initialized at resolution %dx%d", camera_resolution.width, camera_resolution.height);
#endif
                if (ANativeWindow_setBuffersGeometry(app->window, view_width,
                    view_height, WINDOW_FORMAT_RGBA_8888) < 0) {
                    LOGE("Cannot set pixel format!");
                    return;
                }
            }
        	break;
        case APP_CMD_WINDOW_RESIZED:
        	LOGI("APP_CMD_WINDOW_RESIZED");
        	break;
        case APP_CMD_WINDOW_REDRAW_NEEDED:
        	LOGI("APP_CMD_WINDOW_REDRAW_NEEDED");
        	break;
        case APP_CMD_TERM_WINDOW:
        	LOGI("APP_CMD_TERM_WINDOW");
#ifndef BENCHMARK
            engine->capture->release();
#endif
        	break;
        case APP_CMD_INPUT_CHANGED:
        	LOGI("APP_CMD_INPUT_CHANGED");
        	break;
        case APP_CMD_CONTENT_RECT_CHANGED:
        	LOGI("APP_CMD_CONTENT_RECT_CHANGED");
        	break;
        case APP_CMD_CONFIG_CHANGED:
        	LOGI("APP_CMD_CONFIG_CHANGED");
        	break;
        case APP_CMD_LOW_MEMORY:
        	LOGI("APP_CMD_LOW_MEMORY");
        	break;
    }
}


double benchmark(struct Engine* engine, int omp_mode){
    cv::Mat drawing_frame;

	char file_path[256] = {0};
    IplImage  frameImage;
    IplImage* frameImage_hsv;
	IplImage* skinColorSampleImage;
    IplImage* handAreaImage;
    IplImage* handShapeSampleImage;
    IplImage* handShapeSample_gray;

	CvSize screenSize = cvSize(WIDTH, HEIGHT);
	// Create Histgram of Skin Color
	createAssetFile("assets/images/skincolorsample.jpg");
	sprintf(file_path, "%s/%s/%s", DATA_PATH, PACKAGE_NAME, "assets/images/skincolorsample.jpg");
	skinColorSampleImage = cvLoadImage(file_path);
	// Call histgram()
	CvHistogram* skinColorHist;
	double skinColorHist_v_min = 0.0;
	double skinColorHist_v_max = 0.0;
	calcHistgram(skinColorSampleImage, &skinColorHist, &skinColorHist_v_min, &skinColorHist_v_max);

	// Create Histgram of Target Hand Shape
	createAssetFile("assets/images/handshapesample.jpg");
	sprintf(file_path, "%s/%s/%s", DATA_PATH, PACKAGE_NAME, "assets/images/handshapesample.jpg");
	handShapeSampleImage = cvLoadImage(file_path);
	handShapeSample_gray = cvCreateImage(cvGetSize(handShapeSampleImage),IPL_DEPTH_8U,1);
	cvCvtColor(handShapeSampleImage, handShapeSample_gray, CV_BGR2GRAY);
	// Create HoG of Target Hand Shape
	double handShapeSampleFeat[TOTAL_DIM] = {0};
	getHoG(handShapeSample_gray, handShapeSampleFeat);

	char bench_file_path[256] = {0};
	createAssetFile("assets/images/benchmark.jpg");
	sprintf(bench_file_path, "%s/%s/%s", DATA_PATH, PACKAGE_NAME, "assets/images/benchmark.jpg");
	drawing_frame = cvLoadImage(bench_file_path);
	// Detect Skin Area from Capture Image
	frameImage = drawing_frame;
	frameImage_hsv = cvCreateImage(screenSize, IPL_DEPTH_8U, 3);
	cvCvtColor(&frameImage, frameImage_hsv, CV_BGR2HSV);

	CvSeq* convers_ptr = NULL;
	clock_t offset;
	double result;
	if(omp_mode) {
		detectSkinColorAreaOpenMP(frameImage_hsv,
						   &handAreaImage,
							skinColorHist,
						   &convers_ptr,
						   &skinColorHist_v_min,
						   &skinColorHist_v_max);
	} else {
		detectSkinColorArea(frameImage_hsv,
						   &handAreaImage,
							skinColorHist,
						   &convers_ptr,
						   &skinColorHist_v_min,
						   &skinColorHist_v_max);
	}
	// Create HoG of Skin Area
	offset = clock();
	double handAreaFeat[TOTAL_DIM] = {0};
	if(omp_mode) {
		getHoGOpenMP(handAreaImage, handAreaFeat);
	}else{
        getHoG(handAreaImage, handAreaFeat);
	}
	// Calc Distance
	double histDistance = 0.0;
	if(omp_mode) {
		histDistance = getDistanceOpenMP(handShapeSampleFeat, handAreaFeat);
	}else{
		histDistance = getDistance(handShapeSampleFeat, handAreaFeat);
	}
	result = (double)(clock() - offset) / CLOCKS_PER_SEC;
//	char bench_time[256] = {0};
//	sprintf(bench_time, "Result: %lf ms", result);
//	LOGD("%s", bench_time);


	cvReleaseHist(&skinColorHist);
	cvReleaseImage(&handShapeSample_gray);
	cvReleaseImage(&handShapeSampleImage);
	cvReleaseImage(&handAreaImage);
	cvReleaseImage(&skinColorSampleImage);
	cvReleaseImage(&frameImage_hsv);

	return result;
}

void android_main(android_app* app) {

	// Init Parameters
	struct Engine engine;

	// It's magic func for NativeActivityGlue.
	app_dummy();

	// Set UserData
	memset(&engine, 0, sizeof(engine));
	app->userData = &engine;

	app->onAppCmd = engine_handle_cmd;
	engine.app = app;

	// Loop
    char bench_time[256] = {0};
    double result = 0.0;
    LOGD("Processor Num=[%d], Current Thread Num=[%d]", omp_get_num_procs(), omp_get_num_threads());
	while(1) {
        // Read all pending events.
        int ident;
        int events;
        android_poll_source* source;
        // Process system events
        while ((ident=ALooper_pollAll(0, NULL, &events, (void**)&source)) >= 0) {
            // Process this event.
            if (source != NULL) {
                source->process(app, source);
            }

            int i=0;
            result = 0.0;
            for(i = 0; i < REPEATE; i++) {
                result+=benchmark(&engine, FALSE);
            }
            sprintf(bench_time, "Normal Avg(%d) Result: %lf ms",REPEATE ,result/REPEATE);
            LOGD("%s", bench_time);

            result = 0.0;
            for(i = 0; i < REPEATE; i++) {
                result+=benchmark(&engine, TRUE);
            }
            sprintf(bench_time, "Open MP Avg(%d) Result: %lf ms",REPEATE ,result/REPEATE);
            LOGD("%s", bench_time);

            cv::Mat bench_frame = cv::Mat::zeros(640, 480, CV_8UC3);
            cv::putText(bench_frame, std::string(bench_time), cv::Point(300,300),
        			cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0,255,0,255));
            engine_draw_frame(&engine, bench_frame);
        }
	}
}

