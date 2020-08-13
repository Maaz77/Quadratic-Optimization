
default: main.o ssm.o 
	@$(CC)  main.o  ssm.o -o output.o

main.o : main.c
	@$(CC)  -c  main.c
  
ssm.o : SSM.c SSM.h
	@$(CC) -c SSM.c
	
run: default
	@./output.o


clean:
	@$(RM) *.o  
	
	