# Compiler and flags
CXX = g++
CXXFLAGS = -g -O2 -std=c++11

# Include directory for OpenCV
# Adjust this if your OpenCV headers are in a different location
INCLUDES = -I/usr/local/include/opencv4

# Library paths and link directives for OpenCV
# Update these paths according to your OpenCV installation
LDFLAGS = -L/usr/local/lib
LDLIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio

# Name of the executable target
TARGET = face_detector

# Source files
SOURCES = face_detector.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# Rule to link the program
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $(OBJECTS) $(LDLIBS) -o $(TARGET)

# Rule to compile the source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean up
clean:
	rm -f $(TARGET) $(OBJECTS)

# Phony targets
.PHONY: clean
