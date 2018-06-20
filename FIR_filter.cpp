// October 2, 2014
// From: http://www.iowahills.com/A7ExampleCodePage.html
// If you find a problem with this code, please leave us a note on:
// http://www.iowahills.com/feedbackcomments.html

// This is a code implements an FIR filter in floating pt.
// It assumes the max number of taps is 256.
// Reg is the delay registers.

// Remember that these filters have delay, so you need to run the code for M points 
// longer than the number of data points to be filtered in order to get the entire 
// signal through the filter. A reasonable value for M is twice the group delay value.


void RunFIRFilter(double *FiltCoeff, int NumTaps, double *Signal, double *FilteredSignal, int NumSigPts)
{
 int j, k;
 double y, Reg[256];  // This assumes <= 256 taps.

 for(j=0; j<NumTaps; j++)
    Reg[j] = 0.0; // Init the delay registers.

 for(j=0; j<NumSigPts; j++)
     {
        // Shift the register values down and set Reg[0].
     for(k=NumTaps; k>1; k--)Reg[k-1] = Reg[k-2];
        Reg[0] = Signal[j];

      y = 0.0;
      for(k=0; k<NumTaps; k++)y += FiltCoeff[k] * Reg[k];
        FilteredSignal[j] = y;
     }

}

//---------------------------------------------------------------------------

// This code is equivalent to the code above. It should be more efficient because
// it doesn't do a register shift. It rotates the indexes instead.
void RunFIRFilter(double *FirCoeff, int NumTaps, double *Signal, double *FilteredSignal, int NumSigPts)
{
  int j, k, n, Top = 0;
  double y, Reg[256];  // This assumes <= 256 taps.

  for(j=0; j<NumTaps; j++)
      Reg[j] = 0.0;

  for(j=0; j<NumSigPts; j++)
   {
    Reg[Top] = Signal[j];
    y = 0.0;
    n = 0;

    // The FirCoeff index increases while the Reg index decreases.
    for(k=Top; k>=0; k--)
     {
      y += FirCoeff[n++] * Reg[k];
     }
    for(k=NumTaps-1; k>Top; k--)
     {
      y += FirCoeff[n++] * Reg[k];
     }
    FilteredSignal[j] = y;

    Top++;
    if(Top >= NumTaps)Top = 0;
   }
}
