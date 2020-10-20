#include <stdio.h>
#include "hvision.h"

typedef struct {
	int x;
	int y;
} Kernel;


void dilation(IMAGE *, IMAGE *, Kernel *, int);
void erosion(IMAGE *, IMAGE *, Kernel *, int);
void opening(IMAGE *, IMAGE *, Kernel *, int);
void closing(IMAGE *, IMAGE *, Kernel *, int);
void hitAndMiss(IMAGE *, IMAGE *, Kernel *, Kernel *, int, int);

main()
{
	IMAGE *input, *output1, *output2, *output3, *output4, *output5;
	Kernel ker[21], ker1[3], ker2[3];
	
	input=hvReadImage("../lena_binary.im");
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
	output5=hvCopyImage(output1);

	/* set ker*/
	ker[0].x=-2; ker[0].y=-1; // (-2,-1) 
	ker[1].x=-2; ker[1].y= 0; // (-2, 0) 
	ker[2].x=-2; ker[2].y= 1; // (-2, 1) 
	ker[3].x=-1; ker[3].y=-2; // (-1,-2)
	ker[4].x=-1; ker[4].y=-1; // (-1,-1)
	ker[5].x=-1; ker[5].y= 0; // (-1, 0)
	ker[6].x=-1; ker[6].y= 1; // (-1, 1)
	ker[7].x=-1; ker[7].y= 2; // (-1, 2)
	ker[8].x= 0; ker[8].y=-2; // ( 0,-2)
	ker[9].x= 0; ker[9].y=-1; // ( 0,-1)
	ker[10].x= 0; ker[10].y= 0; // ( 0, 0)
	ker[11].x= 0; ker[11].y= 1; // ( 0, 1)
	ker[12].x= 0; ker[12].y= 2; // ( 0, 2)
	ker[13].x= 1; ker[13].y=-2; // ( 1,-2)
	ker[14].x= 1; ker[14].y=-1; // ( 1,-1)
	ker[15].x= 1; ker[15].y= 0; // ( 1, 0)
	ker[16].x= 1; ker[16].y= 1; // ( 1, 1)
	ker[17].x= 1; ker[17].y= 2; // ( 1, 2)
	ker[18].x= 2; ker[18].y=-1; // ( 2,-1)
	ker[19].x= 2; ker[19].y= 0; // ( 2, 0)
	ker[20].x= 2; ker[20].y= 1; // ( 2, 1)

	ker1[0].x= 0; ker1[0].y= 0; // ( 0, 0)
	ker1[1].x= 0; ker1[1].y=-1; // ( 0,-1)
	ker1[2].x= 1; ker1[2].y= 0; // ( 1, 0)


	ker2[0].x=-1; ker2[0].y= 0; // (-1, 0)
	ker2[1].x=-1; ker2[1].y= 1; // (-1, 1)
	ker2[2].x= 0; ker2[2].y= 1; // ( 0, 1)

	
	/* do erosion and opening */
	erosion(input, output1, ker, 21);
	hvWriteImage(output1, "erosion.im");
	
	/* do dilation and closing */
	dilation(input, output2, ker, 21);
	hvWriteImage(output2, "dilation.im");

	/* do opening */
	opening(input, output3, ker, 21);
	hvWriteImage(output3, "opening.im");

	/* do closing */
	closing(input, output4, ker, 21);
	hvWriteImage(output4, "closing.im");
	
	/* do hit and miss to find corner point */
	hitAndMiss(input, output5, ker1, ker2, 3, 3);
	hvWriteImage(output5, "hisAndMiss.im");
	hvFreeImage(output5);
	
}


void dilation(IMAGE *input, IMAGE *output, Kernel *ker,	int num)
{
	int i, j, k, px, py;

	for (i=0; i<input->height; i++) {
		for (j=0; j<input->width; j++) {
			if ( B_PIX(input, i, j) == 1 ) {
				for (k=0; k<num; k++) {
					px=i+ker[k].x;
					py=j+ker[k].y;
					if ( px>=0 && px<input->height && py>=0 && py<input->width )
						B_PIX(output, px, py)=1;
				}
			} 
		}
	}
}

void erosion(IMAGE *input, IMAGE *output, Kernel *ker, int num)
{
	int i, j, k, px, py, Contain;

	for (i=0; i<input->height; i++) {
		for (j=0; j<input->width; j++) {
			Contain=1;
			k=-1;
			while ( Contain && ++k<num ) {
				px=i+ker[k].x; 
				py=j+ker[k].y;
				if ( ( px<0 || px>=input->height || py<0 || py>=input->width )
						|| ( B_PIX(input, px, py)==0 ) )
					Contain=0;
			}
			if (Contain)
				B_PIX(output, i, j)=1;
		}
	}
}


void opening(IMAGE *input, IMAGE *output, Kernel *ker, int num)
{
	IMAGE *temp;
	
	temp=hvCopyImage(output);
	erosion(input, temp, ker, num);
	dilation(temp, output, ker, num);
}

void closing(IMAGE *input, IMAGE *output, Kernel *ker, int num)
{
	IMAGE *temp;
	
	temp=hvCopyImage(output);
	dilation(input, temp, ker, num);
	erosion(temp, output, ker, num);
}	

void hitAndMiss(IMAGE *input, IMAGE *output, Kernel *ker1, Kernel *ker2, int num1, int num2)
{
	IMAGE *complement, *temp1, *temp2;
	int i, j;

	temp1=hvCopyImage(output);
	temp2=hvCopyImage(output);
	/* Make a complement image */ 
	complement=hvCopyImage(output);
	for (i=0; i<input->height; i++)
		for (j=0; j<input->width; j++)
			B_PIX(complement, i, j)=1-B_PIX(input, i, j);


	erosion(input, temp1, ker1, num1);
	erosion(complement, temp2, ker2, num2);

	hvFreeImage(complement);
	
	for (i=0; i<input->height; i++)
		for (j=0; j<input->width; j++)
			if ( (B_PIX(temp1, i, j)==1) && (B_PIX(temp2, i, j)==1) )
				B_PIX(output, i, j)=1;

}
