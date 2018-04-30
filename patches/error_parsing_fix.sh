#!/bin/sh

sed -i 's/org.eclipse.cdt.core.GmakeErrorParser;org.eclipse.cdt.core.GCCErrorParser;org.eclipse.cdt.core.GASErrorParser;org.eclipse.cdt.core.GLDErrorParser;/nvcc.errorParser;org.eclipse.cdt.core.VCErrorParser;org.eclipse.cdt.core.GmakeErrorParser;org.eclipse.cdt.core.MakeErrorParser;org.eclipse.cdt.core.GCCErrorParser;org.eclipse.cdt.core.GASErrorParser;org.eclipse.cdt.core.WorkingDirLocator;org.eclipse.cdt.core.GLDErrorParser;/' $1


