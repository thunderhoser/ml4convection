program read_qpesums_grid

implicit none

integer   :: ii,jj
character :: filename*256
integer   :: yyyy,mm,dd,hh,mn,ss,nx,ny,nz  ! 1-9th vars
character :: proj*4                        ! 10th vars
integer   :: map_scale,projlat1,projlat2,projlon,alon,alat,xy_scale,dx,dy,dxy_scale ! 11-20th vars
integer   :: z_scale,i_bb_mode,unkn01(9)
character :: varname*20,varunit*6
integer   :: var_scale,missing,nradar
integer,allocatable   :: zht(:)
character,allocatable :: mosradar(:,:)
integer*2,allocatable :: var(:,:)
real,allocatable :: var_real(:,:)
real      :: xlat,xlon

call getarg(1,filename)

open(11,file=trim(filename),form='unformatted',status='old',access='stream')
!open(11,file=trim(filename),form='binary',status='old')
read(11) yyyy,mm,dd,hh,mn,ss,nx,ny,nz,proj,&
         map_scale,projlat1,projlat2,projlon,alon,alat,&
         xy_scale,dx,dy,dxy_scale
allocate(var(nx,ny),zht(nz))
allocate(var_real(nx,ny))
read(11) zht,z_scale,i_bb_mode,unkn01,varname,varunit,&
         var_scale,missing,nradar
allocate(mosradar(4,nradar))
read(11) mosradar,var
close(11)
var_real = float(var)/float(var_scale)

do jj = 1,ny
do ii = 1,nx
  xlat = real(alat)/real(xy_scale)+(jj-ny)*real(dy)/real(dxy_scale)
  xlon = real(alon)/real(xy_scale)+(ii-1 )*real(dx)/real(dxy_scale)
  write(*,*)  xlat,xlon,var_real(ii,jj)
end do
end do

deallocate(var,var_real,zht,mosradar)

end
