
# script to produce acoustic params of intensity, F0, F1, F2, F3 every 10 ms  
# command line args give input and output filenames
form acoustic analysis
  text soundname word.wav
endform

intval=0.01
# load soundfile
Read from file... 'soundname$'

# analyse
sound=selected("Sound")
tmin = Get starting time
tmax = Get finishing time
To Pitch... intval 100 500
Rename... pitch

select sound
To Intensity... 75 intval
Rename... intensity

select sound
To Formant (burg)... intval 5 5500 0.025 50
Rename... formants

#printline t   int f0  f1  f2  f3
for i to (tmax-tmin)/0.01
   time = tmin + i * 0.01

   select Pitch pitch
   pitch = Get value at time... time Hertz Linear
   if pitch = undefined
      pitch = -1
   endif

   select Intensity intensity
   intensity = Get value at time... time Cubic
   if intensity = undefined
      intensity = -1
   endif

   select Formant formants
   f1 = Get value at time... 1 time Hertz Linear
   f2 = Get value at time... 2 time Hertz Linear
   f3 = Get value at time... 3 time Hertz Linear
 
   if f1= undefined
      f1= -1
   endif
   if f2= undefined
      f2= -1
   endif
   if f3= undefined
      f3= -1
   endif

  intensity=round(intensity)
  pitch=round(pitch)
  f1=round(f1)
  f2 = round(f2)
  f3 = round(f3)
   printline 'intensity' 'pitch' 'f1' 'f2' 'f3' 
endfor
select Pitch pitch
plus Formant formants
plus Intensity intensity
Remove
