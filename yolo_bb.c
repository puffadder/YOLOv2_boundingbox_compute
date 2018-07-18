#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define OUT_W (13) // width of final layer
#define OUT_H (13) // height of final layer
#define N_BOXES (5) // number of anchor boxes

#define ORIG_W (416)//(768)
#define ORIG_H (416)//(576)
#define NET_W (416)
#define NET_H (416)

#define BOX_STRIDE (OUT_W*OUT_H)

#define NUM_BB (OUT_W*OUT_H*N_BOXES) // total number of possible bounding boxes
//#define NUM_CLASSES (20) // total number of classes for tiny yolo voc
#define NUM_CLASSES (80) // total number of classes for tiny yolo
#define NUM_COORDS (4)
#define NUM_ENTRIES_PER_BOX (NUM_CLASSES+NUM_COORDS+1)
#define NUM_ENTRIES (OUT_W*OUT_H*(N_BOXES*(NUM_CLASSES+NUM_COORDS+1)))

#define PROB_THRESH (0.24) // Default value in darknet
#define NMS_THRESH (0.3) // Default value in darknet

typedef struct
{
    float x, y, w, h;
}box;

typedef struct{
    int index;
    int class;
    float **probs;
} sortable_bbox;

//float biases[10] = {1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52}; // anchor box biases for tiny yolo voc
float biases[10] = {0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828}; // anchor box biases for tiny yolo


void correct_region_boxes(box *boxes, int n, int w, int h, int netw, int neth, int relative);
void do_nms_sort(box *boxes, float **probs, int total, int classes, float thresh);
void draw_detections(int num, float thresh, box *boxes, float **probs, int classes);

float sigmoid(float value)
{
    return 1./(1. + exp(-value));
}

void softmax(float *probs, float *output, int n)
{                                                                      
    int i;                                                             
    float sum = 0;                                                     
    float largest = -FLT_MAX;                                          
    for(i = 0; i < n; ++i){
        if(probs[i] > largest) largest = probs[i];       
    }   
    for(i = 0; i < n; ++i){                                            
        float e = exp(probs[i] - largest);            
        sum += e; 
        output[i] = e;                                          
    }   
    for(i = 0; i < n; ++i){                                            
        output[i] /= sum;                                       
    }   
}

int main()
{
    int status = 0;
    int i, j;
    int row, col;
    int n;
    float *predictions = NULL;
    box *boxes = NULL;
    float **probs = NULL;
    FILE *fp = fopen("outputs.bin", "rb");
    if (!fp)
    {
        printf("Error: Unable to open input file: outputs.bin for reading!\n");
        status = -1;
        goto cleanup;
    }

    predictions = calloc(NUM_ENTRIES, sizeof(float));
    if (!predictions)
    {
        printf("Error: Unable to allocate memory for predictions!\n");
        status = -1;
        goto cleanup;
    }
    if ((NUM_ENTRIES*sizeof(float)) > fread(predictions, 1, NUM_ENTRIES*sizeof(float), fp))
    {
        printf("Error: Unable to read predictions from input file!\n");
        status = -1;
        goto cleanup;
    }
    
    boxes = calloc(NUM_BB, sizeof(box));
    if (!boxes)
    {
        printf("Error: Unable to allocate memory for bounding boxes!\n");
        status = -1;
        goto cleanup;
    }

    probs = calloc(NUM_BB, sizeof(float*));
    if (!probs)
    {
        printf("Unable to allocate memory for output probablities!\n");
        status = -1;
        goto cleanup;
    }
    for (i = 0; i < NUM_BB; i++)
    {
        probs[i] = calloc(NUM_CLASSES+1, sizeof(float));
        if (!probs[i])
        {
            printf("Unable to allocate memory for output probablities!\n");
            status = -1;
            goto cleanup;
        }
    }

    for (i = 0; i < OUT_H*OUT_W; i++)
    {
        row = i / OUT_W;
        col = i % OUT_W;
        for (n = 0; n < N_BOXES; n++)
        {
            int index = n*OUT_W*OUT_H + i;
            for (j = 0; j < NUM_CLASSES; j++)
            {
                probs[index][j] = 0;
            }

            // get region box
            int box_index = i*N_BOXES*NUM_ENTRIES_PER_BOX + n*NUM_ENTRIES_PER_BOX;

            boxes[index].x = (col + sigmoid(predictions[box_index + 0])) / OUT_W;
            boxes[index].y = (row + sigmoid(predictions[box_index + 1])) / OUT_H;
            boxes[index].w = exp(predictions[box_index + 2]) * biases[2*n + 0] / OUT_W;
            boxes[index].h = exp(predictions[box_index + 3]) * biases[2*n + 1] / OUT_H;

            float scale = sigmoid(predictions[box_index + 4]);

#ifdef ENABLE_DEBUG
            printf("\t#obj_index: %d  scale: %f\n", box_index, scale);
            printf("\t#box_index: %d\n", box_index);
            printf("\t#x: %f  y: %f  w: %f  h: %f\n", boxes[index].x, boxes[index].y, boxes[index].w, boxes[index].h);
#endif

            float softmax_preds[NUM_CLASSES];
            softmax(&predictions[box_index + (NUM_COORDS+1)], softmax_preds, NUM_CLASSES);

            float max = 0;
            for (j = 0; j < NUM_CLASSES; j++)
            {
#ifdef ENABLE_DEBUG
                printf("\t\t#class: %d  class_index: %d\n", j, class_index);
#endif

                float prob = scale * softmax_preds[j];

                probs[index][j] = (prob > PROB_THRESH) ? prob : 0;
#ifdef ENABLE_DEBUG
                printf("\t\t\t#prob: %f\n", probs[index][j]);
#endif
                if (prob > max) max = prob;
            }
            probs[index][NUM_CLASSES] = max;
#ifdef ENABLE_DEBUG
            printf("\t#max prob: %f\n", max);
#endif
        }
    }

    correct_region_boxes(boxes, NUM_BB, ORIG_W, ORIG_H, NET_W, NET_H, 1 /*relative*/);
    if (NMS_THRESH) do_nms_sort(boxes, probs, NUM_BB, NUM_CLASSES, NMS_THRESH);

    draw_detections(NUM_BB, PROB_THRESH, boxes, probs, NUM_CLASSES);

cleanup:
    if (probs)
    {
        for (i = 0; i < NUM_BB; i++)
        {
            if (probs[i]) free(probs[i]);
        }
        free(probs);
    }
    if (boxes) free(boxes);
    if (predictions) free(predictions);
    if (fp) fclose(fp);
    return status;
}

void correct_region_boxes(box *boxes, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = boxes[i];
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
#ifdef ENABLE_DEBUG
        printf("\t#new_box x: %f y: %f w: %f h: %f\n", b.x, b.y, b.w, b.h);
#endif
        boxes[i] = b;
    }
}

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}


float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(box a, box b)
{
    return box_intersection(a, b)/box_union(a, b);
}

int nms_comparator(const void *pa, const void *pb)
{
    sortable_bbox a = *(sortable_bbox *)pa;
    sortable_bbox b = *(sortable_bbox *)pb;
    float diff = a.probs[a.index][b.class] - b.probs[b.index][b.class];
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}

void do_nms_sort(box *boxes, float **probs, int total, int classes, float thresh)
{
    int i, j, k;
    sortable_bbox *s = calloc(total, sizeof(sortable_bbox));

    for(i = 0; i < total; ++i){
        s[i].index = i;       
        s[i].class = 0;
        s[i].probs = probs;
    }

    for(k = 0; k < classes; ++k){
        for(i = 0; i < total; ++i){
            s[i].class = k;
        }
        qsort(s, total, sizeof(sortable_bbox), nms_comparator);
        for(i = 0; i < total; ++i){
            if(probs[s[i].index][k] == 0) continue;
            box a = boxes[s[i].index];
            for(j = i+1; j < total; ++j){
                box b = boxes[s[j].index];
                if (box_iou(a, b) > thresh){
                    probs[s[j].index][k] = 0;
                }
            }
        }
    }

#ifdef ENABLE_DEBUG
    printf("\t#Post nms probs:\n");
    for (i=0; i<total; i++)
    {
        for (j=0; j<classes; j++)
        {
            printf("\t\t#[%d][%d]  probs: %f\n", i, j, probs[i][k]);
        }
    }
#endif

    free(s);
}

unsigned int read_class_names(char names[NUM_CLASSES][4096])
{
    char *class_name;
    ssize_t len = 0;
    ssize_t read;
    unsigned int class_num = 0;
    unsigned int is_class_name_available;
    FILE *fp = fopen("coco.names", "r");
    if (!fp)
    {
        printf("Unable to read class name list. Will display label number instead.\n");
        is_class_name_available = 0;
        goto cleanup;
    }
    while ((read = getline(&class_name, &len, fp)) != -1) {
        //printf("Retrieved line of length %zu: %s\n", read, class_name);
        snprintf(names[class_num], read, "%s", class_name);
	names[class_num++][read] = '\0';
    }

    is_class_name_available = 1;

cleanup:
    free(class_name);
    if (fp)
    {
        fclose(fp);
    }
    return is_class_name_available;
}

void draw_detections(int num, float thresh, box *boxes, float **probs, int classes)
{
    int i, j;
    char names[NUM_CLASSES][4096];

    unsigned int is_class_name_available = read_class_names(names);

    for (i = 0; i < num; i++)
    {
        char labelstr[4096] = {0};
        int class = -1;
        for (j = 0; j < classes; j++)
        {
            if (probs[i][j] > thresh)
            {
                if (class < 0)
                {
                    if (is_class_name_available)
                    {
                        strcat(labelstr, names[j]);
                    }
                    else
                    {
                        printf("label #%d, ", j);
                    }
                    class  = j;
                }
                else
                {
                    if (is_class_name_available)
                    {
                        strcat(labelstr, ",");
                        strcat(labelstr, names[j]);
                    }
                    else
                    {
                        printf("label #%d, ", j);
                    }
                }
                if (is_class_name_available)
                {
                    printf("%s: %.0f%%\n", names[j], probs[i][j]*100);
                }
                else
                {
                    printf("\t prob: %f\n", probs[i][j]*100);
                }
            }
        }
        if (class >= 0)
        {
            int width = ORIG_W * 0.006;

            int offset = class*123457 % classes;
            //float red = get_color(2,offset,classes);
            //float green = get_color(1,offset,classes);
            //float blue = get_color(0,offset,classes);
            //float rgb[3];

            //rgb[0] = red;
            //rgb[1] = green;
            //rgb[2] = blue;
            box b = boxes[i];

            int left  = (b.x-b.w/2.)*ORIG_W;
            int right = (b.x+b.w/2.)*ORIG_W;
            int top   = (b.y-b.h/2.)*ORIG_H;
            int bot   = (b.y+b.h/2.)*ORIG_H;

            if(left < 0) left = 0;
            if(right > ORIG_W-1) right = ORIG_W-1;
            if(top < 0) top = 0;
            if(bot > ORIG_H-1) bot = ORIG_H-1;

            printf("\tleft: %d right: %d top: %d bot: %d\n", left, right, top, bot);
        }
    }
}
