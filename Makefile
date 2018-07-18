all:
	gcc -o yolo_bb yolo_bb.c -lm

clean:
	rm -rf *.o yolo_bb
