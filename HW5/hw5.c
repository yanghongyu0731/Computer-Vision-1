#include <stdio.h>
#include "hvision.h"

typedef struct {
	int x;
	int y;
	int value;
} Kernel;


void grayDilation(IMAGE *, IMAGE *, Kernel *, int);
void grayErosion(IMAGE *, IMAGE *, Kernel *, int);
void grayOpening(IMAGE *, IMAGE *, Kernel *, int);
void grayClosing(IMAGE *, IMAGE *, Kernel *, int);


main()
{
	IMAGE *input, *output1, *output2, *output3, *output4;
	Kernel ker[21];
	
	input=hvReadImage("../lena.im");
	if (input == NULL ) {
		printf("cannot open input image\n");
		exit(1);
	}
	output1=hvMakeImage(input->height, input->width, 1, SEQUENTIAL, ONEBYTE);
	if (output1 == I_ERROR) {
		printf("cannot allocate a new image\n");
		exit(1);
	}
	output2=hvCopyImage(output1);
	output3=hvCopyImage(output1);
	output4=hvCopyImage(output1);

	/* set ker*/
	ker[0].x=-2; ker[0].y=-1; ker[0].value=0; // (-2,-1)
	ker[1].x=-2; ker[1].y= 0; ker[1].value=0; // (-2, 0)
	ker[2].x=-2; ker[2].y= 1; ker[2].value=0; // (-2, 1)
	ker[3].x=-1; ker[3].y=-2; ker[3].value=0; // (-1,-2)
	ker[4].x=-1; ker[4].y=-1; ker[4].value=0; // (-1,-1)
	ker[5].x=-1; ker[5].y= 0; ker[5].value=0; // (-1, 0)
	ker[6].x=-1; ker[6].y= 1; ker[6].value=0; // (-1, 1)
	ker[7].x=-1; ker[7].y= 2; ker[7].value=0; // (-1, 2)
	ker[8].x= 0; ker[8].y=-2; ker[8].value=0; // ( 0,-2)
	ker[9].x= 0; ker[9].y=-1; ker[9].value=0; // ( 0,-1)
	ker[10].x= 0; ker[10].y= 0; ker[10].value=0; // ( 0, 0)
	ker[11].x= 0; ker[11].y= 1; ker[11].value=0; // ( 0, 1)
	ker[12].x= 0; ker[12].y= 2; ker[12].value=0; // ( 0, 2)
	ker[13].x= 1; ker[13].y=-2; ker[13].value=0; // ( 1,-2)
	ker[14].x= 1; ker[14].y=-1; ker[14].value=0; // ( 1,-1)
	ker[15].x= 1; ker[15].y= 0; ker[15].value=0; // ( 1, 0)
	ker[16].x= 1; ker[16].y= 1; ker[16].value=0; // ( 1, 1)
	ker[17].x= 1; ker[17].y= 2; ker[17].value=0; // ( 1, 2)
	ker[18].x= 2; ker[18].y=-1; ker[18].value=0; // ( 2,-1)
	ker[19].x= 2; ker[19].y= 0; ker[19].value=0; // ( 2, 0)
	ker[20].x= 2; ker[20].y= 1; ker[20].value=0; // ( 2, 1)


	
	/* do erosion and opening */
	grayErosion(input, output1, ker, 21);
	hvWriteImage(output1, "erosion.im");

	/* do dilation and closing */
	grayDilation(input, output2, ker, 21);
	hvWriteImage(output2, "dilation.im");

	/* do opening */
	grayOpening(input, output3, ker, 21);
	hvWriteImage(output3, "opening.im");

	/* do closing */
	grayClosing(input, output4, ker, 21);
	hvWriteImage(output4, "closing.im");
	
}


void grayDilation(IMAGE *input, IMAGE *output, Kernel *ker,	int num)
{
	int i, j, k, px, py, max, temp;

	for (i=0; i<input->height; i++) {
		for (j=0; j<input->width; j++) {
    	max=0;
      for (k=0; k<num; k++) {
      	px=i-ker[k].x;
        py=j-ker[k].y;
        if ( (px>=0) && (px<input->height) && (py>=0) && (py<input->width) ) {
        	temp=B_PIX(input, px, py)+ ker[k].value;
          if (max<temp)
          	max=temp;
          }
        }
        B_PIX(output, i, j)=max;
		}
	}
}

void grayErosion(IMAGE *input, IMAGE *output, Kernel *ker, int num)
{
	int i, j, k, px, py, min, temp;

	for (i=0; i<input->height; i++) {
		for (j=0; j<input->width; j++) {
			min=256;
      for (k=0; k<21; k++) {
      	px=i+ker[k].x;
        py=j+ker[k].y;
        if ( (px>=0) && (px<input->height) && (py>=0) && (py<input->width) ) {
        	temp=B_PIX( input, px, py)-ker[k].value;
          if (temp<min)
          	min=temp;
        }
      }
      if (min<0)
      	min=0;
      B_PIX(output, i, j)=min;
		}
	}
}


void grayOpening(IMAGE *input, IMAGE *output, Kernel *ker, int num)
{
	IMAGE *temp;
	
	temp=hvCopyImage(output);
	grayErosion(input, temp, ker, num);
	grayDilation(temp, output, ker, num);
}

void grayClosing(IMAGE *input, IMAGE *output, Kernel *ker, int num)
{
	IMAGE *temp;
	
	temp=hvCopyImage(output);
	grayDilation(input, temp, ker, num);
	grayErosion(temp, output, ker, num);
}	

