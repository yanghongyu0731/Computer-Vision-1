#include "hvision.h"
#include <stdio.h>
#include <stdlib.h>
void Yokoi(IMAGE *);

main()
{
	int i, j;
	IMAGE *im, *newIm;

	/* open image */
	im = hvReadImage("../lena_binary.im");
	/* creat a new image */
	newIm=hvMakeImage(im->height/8+2, im->width/8+2, 1, SEQUENTIAL, ONEBYTE);
	if ( newIm == I_ERROR ) {
		printf("cannot open new image\n");
		exit(1);
	}

	/* downsampling */
	for (i=1; i<newIm->height-1; i++)
		for (j=1; j<newIm->width-1; j++) 
			B_PIX(newIm, i, j) = B_PIX(im, (i-1)*8, (j-1)*8);

	/* set border to 0 */
	for (i=0; i<newIm->height; i++) {
		B_PIX(newIm, i, 0)=0;
		B_PIX(newIm, i, newIm->width-1)=0;
	}
	for (j=0; j<newIm->width; j++) {
		B_PIX(newIm, 0, j)=0;
		B_PIX(newIm, newIm->height-1, j)=0;
	}
	hvWriteImage(newIm, "test.im");
	Yokoi(newIm);
}


char h(int b, int c, int d, int e)
{
  if ( b==c )
    if ( b==d && b==e )
      return 'r';
    else
      return 'q';
  else
    return 's';
}


int f(char *a)
{
  int count, i;

  if ( a[0]=='r' && a[1]=='r' && a[2]=='r' && a[3]=='r' )
    return 5;
  else {
    count=0;
    for ( i=0; i<4; i++)
      if ( a[i]=='q' )
        count++;
    return count;
  }
}


void Yokoi(IMAGE *im)
{
  char a[4];
  int **out,i, j;
  FILE *outFile=fopen("output.txt","w");

  out = (int **)malloc(sizeof(int *)*(im->height-2));
  for (i=1; i<im->height-1; i++) {
    out[i] = (int *)malloc(sizeof(int)*(im->width-2));
    for (j=1; j<im->width-1; j++) {
      if ( B_PIX(im, i, j) == 1 ) {
        a[0]=h(B_PIX(im, i, j), B_PIX(im, i, j+1), B_PIX(im, i-1, j+1), B_PIX(im, i-1, j));
        a[1]=h(B_PIX(im, i, j), B_PIX(im, i-1, j), B_PIX(im, i-1, j-1), B_PIX(im, i, j-1));
        a[2]=h(B_PIX(im, i, j), B_PIX(im, i, j-1), B_PIX(im, i+1, j-1), B_PIX(im, i+1, j));
        a[3]=h(B_PIX(im, i, j), B_PIX(im, i+1, j), B_PIX(im, i+1, j+1), B_PIX(im, i, j+1));
        out[i][j]=f(a);
      }else
        out[i][j]=7;
    }
    for (j=1; j<im->width-1; j++) {
      if (out[i][j]==7)
        fprintf(outFile," ");
      else
        fprintf(outFile,"%d",out[i][j]);
    }
    fprintf(outFile,"\n");
  }
  fclose(outFile);
}