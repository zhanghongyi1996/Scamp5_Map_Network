#include <scamp5.hpp>
#include <vector>;
#include <cmath>;
using namespace SCAMP5_PE;
//use R9,R10
void REGISTER_SELECT_WEIGHT_TO_TOP_LEFT(dreg_t loaded_weight, dreg_t target, int weight_index_x, int weight_index_y, int grid, bool duplicate)
{
	scamp5_load_rect(target, weight_index_y * grid, 255 - grid * (weight_index_x + 1) + 1, (weight_index_y + 1) * grid - 1, 255 - grid * weight_index_x);
	scamp5_dynamic_kernel_begin();
	AND(target, target, loaded_weight);
	scamp5_dynamic_kernel_end();
	scamp5_shift(target, - grid * weight_index_x, grid * weight_index_y);
	if (duplicate == 1)
	{
		for (int i = 0; i < log2(256 / grid); i++)
		{
			scamp5_dynamic_kernel_begin();
			MOV(R9, target);
			scamp5_dynamic_kernel_end();
			scamp5_shift(R9, grid * pow(2, i), 0);
			scamp5_dynamic_kernel_begin();
			OR(R10, target, R9);
			MOV(target, R10);
			scamp5_dynamic_kernel_end();
		}

		//duplicate along y axis
		for (int i = 0; i < log2(256 / grid); i++)
		{
			scamp5_dynamic_kernel_begin();
			MOV(R9, target);
			scamp5_dynamic_kernel_end();
			scamp5_shift(R9, 0, -grid * pow(2, i));
			scamp5_dynamic_kernel_begin();
			OR(R10, target, R9);
			MOV(target, R10);
			scamp5_dynamic_kernel_end();
		}
		scamp5_dynamic_kernel_begin();
		CLR(R9, R10);
		scamp5_dynamic_kernel_end();
	}
}


void REGISTER_WEIGHT_IN_GROUP_WITHOUT_DUPLICATE(dreg_t weight_target, const int8_t kernel[][1][16], int grid, int convolution_size)
{
	scamp5_draw_begin(weight_target);
	for (int y = 0; y < convolution_size; y++)
	{
		for (int x = 0; x < convolution_size; x++)
		{
			for (int group = 0; group < grid * grid; group++)
			{
				int weight = kernel[group][0][y * convolution_size + x];
				int groupx = group % grid;
				int groupy = group / grid;
				if (weight == 1)
				{
					scamp5_draw_point(grid * y + groupy, 255 - grid * x - groupx);
				}
			}
		}
	}
	scamp5_draw_end();
}

//use R10, R8, R9, R3
void REGISTER_IMAGE_IN_GROUP(areg_t input, int grid)
{
	scamp5_load_pattern(R8, 0, -1, 255 - grid + 1, 255 - grid + 1);
	scamp5_dynamic_kernel_begin();
	all();
	res(E);
	WHERE(R8);
	mov(E, input);
	all();
	res(input);
	mov(input, E);
	MOV(R9, R8);
	scamp5_dynamic_kernel_end();
	//duplicate along x axis
	for (int i = 0; i < log2(grid); i++)
	{
		for (int cg = 0; cg < pow(2, i); cg++)
		{
			scamp5_dynamic_kernel_begin();
			bus(NEWS, E);
			bus(E, XW);
			scamp5_dynamic_kernel_end();
		}
		scamp5_shift(R9, pow(2,i), 0);
		scamp5_dynamic_kernel_begin();
		WHERE(R9);
		mov(input,E);
		all();
		OR(R10, R8, R9);
		MOV(R8, R10);
		MOV(R9, R10);
		mov(E, input);
		scamp5_dynamic_kernel_end();
	}

	//duplicate along y axis
	for (int i = 0; i < log2(grid); i++)
	{
		for (int cg = 0; cg < pow(2, i); cg++)
		{
			scamp5_dynamic_kernel_begin();
			bus(NEWS, E);
			bus(E, XN);
			scamp5_dynamic_kernel_end();
		}
		scamp5_shift(R9, 0, -pow(2, i));
		scamp5_dynamic_kernel_begin();
		WHERE(R9);
		mov(input, E);
		all();
		OR(R10, R8, R9);
		MOV(R8, R10);
		MOV(R9, R10);
		mov(E, input);
		scamp5_dynamic_kernel_end();
	}
	scamp5_dynamic_kernel_begin();
	CLR(R8, R9, R10, R3);
	scamp5_dynamic_kernel_end();
}

//use R8, R9, R3
void REGISTER_WEIGHT_IN_GROUP(dreg_t weight_target, const int8_t kernel[][1][16], int grid, int convolution_size)
{
	scamp5_draw_begin(weight_target);
    for (int y = 0; y < convolution_size; y++)
	{
		for (int x = 0; x < convolution_size; x++)
		{
			for (int group = 0; group < grid * grid; group++)
			{
				int weight = kernel[group][0][y * convolution_size + x];
				int groupx = group % grid;
				int groupy = group / grid;
				if (weight == 1)
				{
					scamp5_draw_point(grid * y + groupy, 255 - grid * x - groupx);
				}
			}
		}
	}
	scamp5_draw_end();

	//duplicate along x axis
	int gap = grid * convolution_size;
	for (int i = 0; i < log2(256 / gap); i++)
	{
		scamp5_dynamic_kernel_begin();
		MOV(R8, weight_target);
		MOV(R9, weight_target);
		scamp5_dynamic_kernel_end();
		scamp5_shift(R8, gap * pow(2, i), 0);
		scamp5_dynamic_kernel_begin();
		OR(weight_target, R8, R9);
		scamp5_dynamic_kernel_end();
	}

	//duplicate along y axis
	for (int i = 0; i < log2(256 / gap); i++)
	{
		scamp5_dynamic_kernel_begin();
		MOV(R8, weight_target);
		MOV(R9, weight_target);
		scamp5_dynamic_kernel_end();
		scamp5_shift(R8, 0, -gap * pow(2, i));
		scamp5_dynamic_kernel_begin();
		OR(weight_target, R8, R9);
		scamp5_dynamic_kernel_end();
	}
	scamp5_kernel_begin();
	CLR(R8, R9, R3);
	scamp5_kernel_end();
}


void REGISTER_FC_WEIGHT(dreg_t weight_target, const int8_t kernel[][1024], int grid, int pixel_size, int output_size)
{
	int num_neuron_max = 256 / (pixel_size * grid);
	scamp5_draw_begin(weight_target);
	for (int step = 0; step < output_size; step++)
	{
		for (int channel = 0; channel < grid * grid; channel++)
		{
			for (int y = 0; y < pixel_size; y++)
			{
				for (int x = 0; x < pixel_size; x++)
				{
					int weight = kernel[step][channel * pixel_size * pixel_size + y * pixel_size + x];
					int neuron_pos_shift_x = step % num_neuron_max;
					int neuron_pos_shift_y = step / num_neuron_max;
					if (weight == 1)
					{
						scamp5_draw_point(neuron_pos_shift_y * grid + (256 / pixel_size * y + channel / grid), (255 - 256 / pixel_size * x - channel % grid) - neuron_pos_shift_x * grid);
					}
				}
			}
		}
	}
	scamp5_draw_end();
}


void CONV_FOLD(areg_t reg, int log2_box_size, int grid)
{
	for (int x = 0; x < log2_box_size; x++)
	{
		//DIVIDE IF THERE IS ANOTHER STEP 
		if (x != log2_box_size)
		{
			scamp5_dynamic_kernel_begin();
			divq(E, reg);
			mov(reg, E);
			scamp5_dynamic_kernel_end();
		}
		//STEP 2, 4, 8...
		for (int n = 0; n < pow(2, x) * grid; n++)
		{
			scamp5_kernel_begin();
			bus(NEWS, E);
			bus(E, XE);
			scamp5_kernel_end();
		}
		
		scamp5_load_rect(R8, 0, 0, 255, pow(2, x) * grid - 1);
		scamp5_kernel_begin();
		WHERE(R8);
		res(E);
		all();
		CLR(R8);
		scamp5_kernel_end();
		
		//ACCUMULATE 
		scamp5_kernel_begin();
		add(F, F, E);
		scamp5_kernel_end();

	}

	for (int y = 0; y < log2_box_size; y++)
	{
		//DIVIDE IF THERE IS ANOTHER STEP 
		if (y != log2_box_size)
		{
			scamp5_kernel_begin();
			divq(E, F);
			mov(F, E);
			scamp5_kernel_end();
		}
		//STEP 2, 4, 8...
		for (int n = 0; n < pow(2, y) * grid; n++)
		{
			scamp5_kernel_begin();
			bus(NEWS, E);
			bus(E, XS);
			scamp5_kernel_end();
		}
		
		scamp5_load_rect(R8, 255 - pow(2, y) * grid + 1, 0, 255, 255);
		scamp5_kernel_begin();
		WHERE(R8);
		res(E);
		all();
		CLR(R8);
		scamp5_kernel_end();
		
		//ACCUMULATE 
		scamp5_kernel_begin();
		add(F, F, E);
		scamp5_kernel_end();
	}
	scamp5_kernel_begin();
	res(E);
	scamp5_kernel_end();
}

//Use R8,E,F
void FOLD_CONV_IN_GROUP(areg_t input, dreg_t weight, areg_t target, int group_num, int convolution_size, int stride)
{
	int grid = sqrt(group_num);
	scamp5_dynamic_kernel_begin();
	all();
	res(target);
	scamp5_dynamic_kernel_end();
	for (int x = 0; x < convolution_size / stride; x++)
	{
		for (int y = 0; y < convolution_size / stride; y++)
		{
			scamp5_dynamic_kernel_begin();
			neg(F, input);
			MOV(R8, weight);
			if (x > 0 | y > 0)
			{
				scamp5_shift(R8, x * stride * grid, -y * stride * grid);
			}
			WHERE(R8);
			mov(F, input);
			all();
			CLR(R8);
			scamp5_dynamic_kernel_end();
			CONV_FOLD(F, log2(convolution_size), grid);
			scamp5_load_pattern(R8, - grid * y * stride, - 1 - grid * x * stride, 255 - convolution_size * grid + grid, 255 - convolution_size * grid + grid);
			scamp5_dynamic_kernel_begin();
			WHERE(R8);
			mov(target, F);
			all();
			CLR(R8);
			res(F);
			scamp5_dynamic_kernel_end();
		}
	}
}



//USE R8,R9,R10,F,E
void CONV_IN_GROUP_SNAKE_PATH(areg_t input, dreg_t weight, areg_t target, int group_num, int convolution_size)
{
	int counter = convolution_size * convolution_size;
	int grid = sqrt(group_num);
	int x, y;
	for (int i = 0; i < counter; i++)
	{
		y = i / convolution_size;
		x = i % convolution_size;
		if (y % 2 == 1)
		{
			x = convolution_size - i % convolution_size - 1;
		}
		REGISTER_SELECT_WEIGHT_TO_TOP_LEFT(weight, R8, x, y, grid, 1);
		scamp5_dynamic_kernel_begin();
		neg(F, input);
		WHERE(R8);
		mov(F, input);
		all();
		//div advance
		for (int dc = 0; dc < log2(convolution_size); dc++)
		{
			divq(E, F);
			mov(F, E);
		}
		scamp5_dynamic_kernel_end();
	}
}

//USE R8
void RELU_IN_SCAMP(areg_t target)
{
	scamp5_dynamic_kernel_begin();
	where(target);
	MOV(R8,FLAG);
	NOT(R8);
	WHERE(R8);
	res(target);
	all();
	CLR(R8);
	scamp5_dynamic_kernel_end();
}

//USE R8,A,E,F
void MAXPOOLING_SCAMP(areg_t input, int pooling_len, int last_stride, int group)
{
	int grid = sqrt(group);
	scamp5_dynamic_kernel_begin();
	mov(F, input);
	mov(E, F);
	scamp5_dynamic_kernel_end();
	//max_pooling along x axis
	for (int i = 0; i < log2(pooling_len); i++)
	{
		for (int step = 0; step < last_stride * grid * pow(2, i); step++)
		{
			scamp5_dynamic_kernel_begin();
			bus(NEWS, E);
			bus(E, XE);
			scamp5_dynamic_kernel_end();
		}
		scamp5_load_pattern(R8, 0, -1, 255 - last_stride * grid + grid, 255 - last_stride * grid * pow(2, i+1) + grid);
		scamp5_dynamic_kernel_begin();
		sub(A, E, F);
		where(A);
		mov(F, E);
		all();
		res(E);
		WHERE(R8);
		mov(E, F);
		all();
		mov(F, E);
		scamp5_dynamic_kernel_end();
	}

	//max_pooling along y axis
	for (int i = 0; i < log2(pooling_len); i++)
	{
		for (int step = 0; step < last_stride * grid * pow(2, i); step++)
		{
			scamp5_dynamic_kernel_begin();
			bus(NEWS, E);
			bus(E, XS);
			scamp5_dynamic_kernel_end();
		}
		scamp5_load_pattern(R8, 0, -1, 255 - last_stride * grid * pow(2, i + 1) + grid, 255 - last_stride * grid * pow(2, log2(pooling_len)) + grid);
		scamp5_dynamic_kernel_begin();
		sub(A, E, F);
		where(A);
		mov(F, E);
		all();
		res(E);
		WHERE(R8);
		mov(E, F);
		all();
		mov(F, E);
		scamp5_dynamic_kernel_end();
	}
	scamp5_dynamic_kernel_begin();
	mov(input, E);
	res(A);
	res(E, F);
	CLR(R8);
	scamp5_dynamic_kernel_end();
}

int FC_SCAMP_SINGLE_OUTPUT(areg_t input, dreg_t weight, int pixel_len, int pooling_len, int grid, int output_neuron_index)
{
	int gap = (256 / pixel_len) * pooling_len;
	int max_len_per_square = gap / grid;
	scamp5_load_pattern(R8, 0, -1, 255 - gap + grid, 255 - gap + grid);
	scamp5_dynamic_kernel_begin();
	MOV(R9, weight);
	scamp5_dynamic_kernel_end();
	int shift_x = - grid * (output_neuron_index % max_len_per_square);
	int shift_y = grid * (output_neuron_index / max_len_per_square);
	if (shift_x != 0 | shift_y != 0)
	{
		scamp5_shift(R9, shift_x, shift_y);
	}
	scamp5_dynamic_kernel_begin();
	AND(R9, R9, R8);
	scamp5_dynamic_kernel_end();
	scamp5_dynamic_kernel_begin();
	neg(F, input);
	WHERE(R9);
	mov(F, input);
	all();
	CLR(R8, R9);
	scamp5_dynamic_kernel_end();

	//scamp5_load_pattern(R8, 0, -1, 256 / 8 - 1 - gap + grid, 256 / 8 - 1 - gap + grid);
	/*
	int sum = 0;
	int result = 0;
	for (int i = 0; i < 4; i++)
	{
		int x = i % 2;
		int y = i / 2;
		result = scamp5_global_sum_sparse(F, 0 + y * 128, -1 + x * 128, 256 / 2 - 1 - gap + grid, 256 / 2 - 1 - gap + grid);
		sum = sum + result;
	}
	//std::cout << result << std::endl;
	*/
	int result = scamp5_global_sum_sparse(F, 0, -1, 256 - 1 - gap + grid, 256 - 1 - gap + grid);
	return result;
}