CXX=g++
CXXFLAGS=-std=c++11 -Wall -Wextra -O2
LDFLAGS=
OBJ_FILES=process_sass_dir.o
TARGET=process_sass_dir

all: $(TARGET)

$(TARGET): $(OBJ_FILES)
	$(CXX) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ_FILES) $(TARGET)

.PHONY: all clean
