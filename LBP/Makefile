# Compiler and flags
CXX = g++
CXXFLAGS = -g -std=c++11

# Name of the executable target
TARGET = face_detector

# Source files
SOURCES = lbp.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# Rule to link the program
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(TARGET) `pkg-config --cflags --libs opencv4`

# Rule to compile the source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@ `pkg-config --cflags --libs opencv4`

# Clean up
clean:
	rm -f $(TARGET) $(OBJECTS)

# Phony targets
.PHONY: clean
