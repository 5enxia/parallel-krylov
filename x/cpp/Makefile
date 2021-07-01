CC := g++ -std=c++11
OPT := -Wall -Wextra -march=native -O2

BLASINCLUDE := -I/usr/local/opt/openblas/include
BLASLIB := -L/usr/local/opt/openblas/lib/ -lopenblas
APPLEBLAS := -I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers/ -Framework Accelerate

applecg:
	$(CC) $(OPT) cg.cpp -lcnpy -I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers/ -Framework Accelerate -o cg.out
cg: cg
	$(CC) $(OPT) cg.cpp -lcnpy $(BLASINCLUDE) $(BLASLIB) -o cg.out
kcg: kskipcg
	$(CC) $(OPT) kskipcg.cpp -lcnpy $(BLASINCLUDE) $(BLASLIB) -o kskipcg.out
kmrr: kskipmrr
	$(CC) $(OPT) kskipmrr.cpp -lcnpy $(BLASINCLUDE) $(BLASLIB) -o kskipmrr.out
