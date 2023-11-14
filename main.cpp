/*************************************************************************
 * SCAMP Vision Chip Development System Library
 *------------------------------------------------------------------------
 * Copyright (c) 2020 The University of Manchester. All Rights Reserved.
 *
 *************************************************************************/
/*
* Scamp5d M0 Example - Simulation
*
* Note: this source code works on both of the hardware and the simulation 
*       but with an emphasis on demonstrating simulation-related functions.
*/

#include <scamp5.hpp>
#include <iostream>
#include <string.h>
#include <cmath>
#include "CONVOLUTION_FUNCS.hpp"
#include "weights.hpp"
#include "new_conv.hpp"
using namespace SCAMP5_PE;

using namespace std;

volatile int threshold;



int main(){

    vs_sim::config("server_ip","127.0.0.1");
    vs_sim::config("server_port","27715");
    
    // Initialization
    vs_init();

    vs_sim::enable_keyboard_control();// this allow a few shortcuts to be used. e.g. 'Q' to quit. 
	vs_sim::reset_model(3);// reset model is also used to configure the error model


    // Setup Host GUI
	vs_gui_set_info(VS_M0_PROJECT_INFO_STRING);

    auto display_1 = vs_gui_add_display("capture",0,0);
    auto display_2 = vs_gui_add_display("digital",0,1);

    auto slider_threshold = vs_gui_add_slider("threshold: ",-100,100,0,&threshold);
	int grid = 4;
	int row;
	int column;
	REGISTER_WEIGHT_IN_GROUP(R7, weights, 4, 4);
	REGISTER_FC_WEIGHT(R6, fc_weights, 4, 8, 16);
	//REGISTER_WEIGHT_IN_GROUP_WITHOUT_DUPLICATE(R7, weights, 4, 4);
	//DRAW_ALL_WEIGHT(R7, 4, weights, 16);

	while (1) {
		vs_frame_loop_control();
		scamp5_kernel_begin();
		all();
		get_image(C, D);
		res(D);
		scamp5_kernel_end();
		
		REGISTER_IMAGE_IN_GROUP(C, 4);
		FOLD_CONV_IN_GROUP(C, R7, D, 16, 4, 2);
		
		//RELU_IN_SCAMP(D);
		
		MAXPOOLING_SCAMP(D, 4, 2, 16);
		//int fc_result = FC_SCAMP_SINGLE_OUTPUT(D, R6, 32, 4, 4, 13);
		//cout << fc_result << endl;
	
		int sums[] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
		int max = 0;
		int min = 200;
		for (int i = 0; i < 16; i++)
		{
			int fc_result = FC_SCAMP_SINGLE_OUTPUT(D, R6, 32, 4, 4, i);
			sums[i] = fc_result;
			if (fc_result > max)
			{
				max = fc_result;
			}
			if (fc_result < min)
			{
				min = fc_result;
			}
		}
		int final_result[] = { 0,0,0 };
		for (int place = 0; place < 3; place++)
		{
			for (int i = 0; i < 16; i++)
		{
			final_result[place] = final_result[place] + fc_weights_2[place][i] * (sums[i] - min);
		}
			cout << final_result[place] << ',';
		}
		cout << endl;

	}

    return 0;
}
