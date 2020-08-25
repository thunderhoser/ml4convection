program test

integer :: ii,jj
character :: filename*256
character*1 :: btmp(921,881)
real :: xlat(921,881),xlon(921,881)

call getarg(1,filename)

open(12,file="./Proj_NOAA.GSD",form="unformatted",access="stream",status="old")
read(12) xlat,xlon
close(12)
open(11,file=trim(filename),form="unformatted",access="stream",status="old")
read(11) btmp
close(11)

do jj = 1,881
do ii = 1,921
  write(*,*) xlat(ii,jj),xlon(ii,jj),ichar(btmp(ii,jj))
end do
end do

end program
