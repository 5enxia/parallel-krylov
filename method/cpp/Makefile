LDFLAGS := -L/usr/local/opt/openblas/lib/
CPPFLAG := -I/usr/local/opt/openblas/include

tmp:
	g++ -std=c++11 -Wall -Wextra -march=native -O2 cg.cpp -lcnpy -I/usr/local/opt/openblas/include/ -L/usr/local/opt/openblas/lib -lopenblas -o cg
cg: cg
	g++ -std=c++11 -Wall -Wextra -march=native -O2 cg.cpp -lcnpy -I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers/ -Framework Accelerate -o cg
kcg: kskipcg
	g++ -std=c++11 -Wall -Wextra -march=native -O2 kskipcg.cpp -lcnpy -I/usr/local/opt/openblas/include/ -L/usr/local/opt/openblas/lib -lopenblas -o kcg
kmrr: kskipmrr
	g++ -std=c++11 -Wall -Wextra -march=native -O2 kskipmrr.cpp -lcnpy -I/usr/local/opt/openblas/include/ -L/usr/local/opt/openblas/lib -lopenblas -o kmrr
