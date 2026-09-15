CXX = g++
CPPFLAGS = -g -O2 -std=c++20

PROJDIR = $(shell ( cd $$CMSSW_BASE/src && /bin/pwd ))
BOOST_BASE = $(shell ( cd $(PROJDIR) && scram tool info boost ) | grep -e '^BOOST_BASE' | sed 's/.*=//')
ONNX_BASE = $(shell ( cd $(PROJDIR) && scram tool info onnxruntime ) | grep -e '^ONNXRUNTIME_BASE' | sed 's/.*=//')

ROOTFLAGS = $(shell root-config --libs --cflags --auxcflags)
ROOTFLAGS += -lMLP -lMinuit -lTreePlayer -lTMVA -lXMLIO  -lMLP -lm -lTMVAUtils

CORRECTIONLIBFLAGS = -I $(shell correction config --incdir --ldflags --rpath)

ONNX_FLAGS = -I $(ONNX_BASE)/include -L $(ONNX_BASE)/lib -lonnxruntime
BOOST_FLAGS = -I $(BOOST_BASE)/include -L $(BOOST_BASE)/lib

MY_FLAGS = -I include/ -I $(PROJDIR) $(ONNX_FLAGS) $(BOOST_FLAGS) $(ROOTFLAGS) $(CORRECTIONLIBFLAGS)

MAKEFLAGS := --jobs=8

SRCS = $(wildcard src/*.cpp)
OBJS = $(patsubst src/%.cpp,build/%.o,$(SRCS))
DEPS = $(OBJS:.o=.d)

# -MP adds phony targets for the listed headers, so removing one does not break the build.
DEPFLAGS = -MMD -MP

TARGET = runAnalysis

BUILD_DIR = build/
BIN_DIR = bin

all: main

main: $(OBJS) | $(BIN_DIR)
	$(CXX) $(CPPFLAGS) $(MY_FLAGS) $(OBJS) -o $(BIN_DIR)/$(TARGET)

build/%.o: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) $(MY_FLAGS) -c $< -o $@

# Rebuild an object when any header it includes changes.
-include $(DEPS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

.DELETE_ON_ERROR:


clean:
	$(RM) $(OBJS) $(DEPS) bin/$(TARGET)

clearplots:
	$(RM)

.PHONY: all main clean clearplots
