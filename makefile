JFLAGS = -d .
JC = javac
.SUFFIXES: .java .class
.java.class:
		$(JC) $(JFLAGS) $*.java

CLASSES = \
		Network/Layer.java \
		Network/Net.java \
		Main.java

default: classes

classes: $(CLASSES:.java=.class)

clean:
		$(RM) *.class
