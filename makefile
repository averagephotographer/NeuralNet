JFLAGS = -d .
JC = javac
.SUFFIXES: .java .class
.java.class:
		$(JC) $(JFLAGS) $*.java

CLASSES = \
		Array.java \
		MiniBatch.java \
		Net.java

default: classes

classes: $(CLASSES:.java=.class)

clean:
		$(RM) *.class
