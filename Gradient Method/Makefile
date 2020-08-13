all: default

default: gradient_method 

gradient_method: main.c Makefile
	@$(CC)  -framework Accelerate -o main.o  main.c
  

run: gradient_method
	@./main.o


clean:
	$(RM) *.o  
	
	