import { Component, OnInit } from '@angular/core';
import { UserService } from '../services/user.service';

@Component({
  selector: 'ngx-adduser',
  templateUrl: './adduser.component.html',
  styleUrls: ['./adduser.component.scss'],
  providers: [UserService]
})
export class AdduserComponent implements OnInit {

  username: string = '';
  status: string = 'active';

  constructor(private userService: UserService) { }

  ngOnInit(): void {
  }

  AddUser(){
    this.userService.AddNewUser(this.username,this.status);
    console.log(this.userService.users);
  }

}
